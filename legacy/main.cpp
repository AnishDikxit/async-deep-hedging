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
#include <algorithm>

struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
    long long execution_time; 
    long long expiration_time; // NEW: The Time-To-Live timestamp
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
                if(new_order.id != 999999 && asks[i].id!=999999)continue;
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
                if(new_order.id != 999999 && bids[i].id!=999999)continue;
                bids.erase(bids.begin() + i);
                return; 
            }
        }
        asks.push_back(new_order); 
    }
}

void cleanup_expired_orders(long long current_time) {
    // 1. Purge expired Bids
    bids.erase(
        std::remove_if(bids.begin(), bids.end(),
            [current_time](const Order& o) { 
                // Return true if the order has passed its expiration time
                return o.expiration_time != 0 && o.expiration_time <= current_time; 
            }),
        bids.end()
    );

    // 2. Purge expired Asks
    asks.erase(
        std::remove_if(asks.begin(), asks.end(),
            [current_time](const Order& o) { 
                return o.expiration_time != 0 && o.expiration_time <= current_time; 
            }),
        asks.end()
    );
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
    // --- HAWKES PROCESS PARAMETERS ---
    double base_mu = 0.02;           // 2% chance of normal background activity
    double hawkes_excitation = 0.0;  // The 'Panic' multiplier
    double alpha_jump = 0.15;        // How much panic is added per event
    double decay_rate = 0.98;        // How fast the market calms down

    while (true) {
        // --- 1. CHECK FOR INCOMING ORDERS FROM PYTHON (MOVED TO TOP) ---
        redisReply *reply = (redisReply*)redisCommand(c, "LPOP incoming_orders");
        if (reply->type == REDIS_REPLY_STRING) {
            std::string order_str = reply->str;
            if (order_str == "RESET") {
                bids.clear();
                asks.clear();
                current_market_price = 100; 
                hawkes_excitation = 0.0; 
                latency_queue = std::priority_queue<Order, std::vector<Order>, CompareOrder>();
                // std::cout << "[EXCHANGE] Episode Reset! Price Anchored to $100.\n";
                // NEW: Force-publish the clean price immediately so Python doesn't grab a stale tick!
                redisReply *pub_reply = (redisReply*)redisCommand(c, "PUBLISH market_data %d", current_market_price);
                freeReplyObject(pub_reply);
            } else {
                std::stringstream ss(order_str);
                int price, is_buy_int;
                if (ss >> price >> is_buy_int) {
                    Order new_order;
                    new_order.id = order_id_counter++;
                    new_order.quantity = 10;
                    new_order.price = price;
                    new_order.is_buy = (is_buy_int == 1);
                    new_order.execution_time = current_time + 50;
                    new_order.expiration_time = 0;
                    latency_queue.push(new_order);
                }
            }
        }
        freeReplyObject(reply);

        // --- 2. INCREMENT TIME ---
        current_time++;

        // --- 3. SIMULATE HAWKES PROCESS MARKET MOVEMENT ---
        hawkes_excitation *= decay_rate; 
        double current_lambda = base_mu + hawkes_excitation;

        if ((rand() % 1000) / 1000.0 < current_lambda) {
            hawkes_excitation += alpha_jump;
            current_market_price += ((rand() % 5) - 2); 
            if (current_market_price < 1) current_market_price = 1;
            
            int burst_orders = 2 + (rand() % 4); 
            for (int i = 0; i < burst_orders; i++) {
                Order dummy_order;
                dummy_order.id = 999999; 
                dummy_order.quantity = 10;
                dummy_order.is_buy = (rand() % 2 == 0);
                int spread = (rand() % 3) + 1;
                dummy_order.price = dummy_order.is_buy ? (current_market_price - spread) : (current_market_price + spread);
                dummy_order.expiration_time = current_time + 50 + (rand() % 250);
                add_order(dummy_order);
            }
            
            // Because this happens AFTER the reset check, Python is guaranteed to get the clean $100 price!
            redisReply *pub_reply = (redisReply*)redisCommand(c, "PUBLISH market_data %d", current_market_price);
            freeReplyObject(pub_reply);
        }

        // --- 4. PROCESS LATENCY QUEUE ---
        while (!latency_queue.empty() && latency_queue.top().execution_time <= current_time) {
            Order ready_order = latency_queue.top();
            latency_queue.pop();
            add_order(ready_order); 
        }

        // --- 5. SWEEP THE ORDER BOOK ---
        cleanup_expired_orders(current_time);
    }
    redisFree(c);
    return 0;
}