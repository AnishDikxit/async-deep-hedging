#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <chrono>
#include <string>
#include <sstream>
#include <hiredis/hiredis.h> // NEW: The Redis C++ Library

struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
    long long execution_time; 
};

struct CompareOrder {
    bool operator()(Order const& o1, Order const& o2) {
        return o1.execution_time > o2.execution_time; 
    }
};

std::vector<Order> bids; 
std::vector<Order> asks;

void add_order(Order new_order) {
    if (new_order.is_buy) {
        for (int i = 0; i < asks.size(); i++) {
            if (asks[i].price <= new_order.price) {
                // std::cout << "   -> [MATCH] Buyer " << new_order.id 
                //      << " bought from Seller " << asks[i].id << " at $" << asks[i].price << "\n\n";
                asks.erase(asks.begin() + i);
                return; 
            }
        }
        bids.push_back(new_order); 
    } else {
        for (int i = 0; i < bids.size(); i++) {
            if (bids[i].price >= new_order.price) {
                // std::cout << "   -> [MATCH] Seller " << new_order.id 
                //      << " sold to Buyer " << bids[i].id << " at $" << bids[i].price << "\n\n";
                bids.erase(bids.begin() + i);
                return; 
            }
        }
        asks.push_back(new_order); 
    }
}

int main() {
    srand(time(0)); 
    
    // 1. CONNECT TO REDIS
    redisContext *c = redisConnect("127.0.0.1", 6379);
    if (c == NULL || c->err) {
        if (c) { std::cout << "Redis Error: " << c->errstr << "\n"; redisFree(c); } 
        else { std::cout << "Cannot allocate Redis context\n"; }
        return 1;
    }
    std::cout << "--- Successfully Connected to Redis ---\n";

    std::priority_queue<Order, std::vector<Order>, CompareOrder> latency_queue;
    long long current_time = 0; 
    int order_id_counter = 1;
    int current_market_price = 100;

    std::cout << "--- Starting Async Market Simulator ---\n";

    while (true) {
        current_time++;

        // 2. SIMULATE BACKGROUND MARKET MOVEMENT
        if (rand() % 100 < 10) { // 10% chance price moves slightly
            current_market_price += ((rand() % 3) - 1); 
            
            // BROADCAST NEW PRICE TO PYTHON
            redisReply *pub_reply = (redisReply*)redisCommand(c, "PUBLISH market_data %d", current_market_price);
            freeReplyObject(pub_reply);
        }

        // 3. CHECK FOR INCOMING ORDERS FROM PYTHON (Non-blocking)
        // We look for a string formatted like: "PRICE IS_BUY" (e.g., "102 1" for Buy at $102)
        redisReply *reply = (redisReply*)redisCommand(c, "LPOP incoming_orders");
        
        if (reply->type == REDIS_REPLY_STRING) {
            std::string order_str = reply->str;
            std::stringstream ss(order_str);
            int price, is_buy_int;
            
            // Parse the string into an Order
            if (ss >> price >> is_buy_int) {
                Order new_order;
                new_order.id = order_id_counter++;
                new_order.quantity = 10;
                new_order.price = price;
                new_order.is_buy = (is_buy_int == 1);
                
                // ADD 50ms LATENCY
                new_order.execution_time = current_time + 50; 
                latency_queue.push(new_order);
                
                // std::cout << "[t=" << current_time << "] Network Received Order: " 
                //      << (new_order.is_buy ? "BUY" : "SELL") << " at $" << price 
                //      << " (Delayed to t=" << new_order.execution_time << ")\n";
            }
        }
        freeReplyObject(reply);

        // 4. PROCESS LATENCY QUEUE
        while (!latency_queue.empty() && latency_queue.top().execution_time <= current_time) {
            Order ready_order = latency_queue.top();
            latency_queue.pop();
            
            // std::cout << "[t=" << current_time << "] RELEASING delayed order " << ready_order.id << " into book!\n";
            add_order(ready_order); 
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Run fast, but don't burn the CPU
    }

    redisFree(c);
    return 0;
}