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
    // --- HAWKES PROCESS PARAMETERS ---
    double base_mu = 0.02;           // 2% chance of normal background activity
    double hawkes_excitation = 0.0;  // The 'Panic' multiplier
    double alpha_jump = 0.15;        // How much panic is added per event
    double decay_rate = 0.98;        // How fast the market calms down
    // 2. SIMULATE HAWKES PROCESS MARKET MOVEMENT
        
    // A. Decay the panic from the previous millisecond
    hawkes_excitation *= decay_rate; 
    double current_lambda = base_mu + hawkes_excitation;

    // B. Roll the dice to see if a market event happens right now
    if ((rand() % 1000) / 1000.0 < current_lambda) {
        
        // 1. Spike the excitation (The Self-Exciting property)
        hawkes_excitation += alpha_jump;

        // 2. Shift the mid-price based on the volatility
        current_market_price += ((rand() % 5) - 2); 
        
        // 3. Flood the Order Book with dummy liquidity
        // This creates the Execution Risk that punishes the AI's latency
        int burst_orders = 2 + (rand() % 4); 
        for (int i = 0; i < burst_orders; i++) {
            Order dummy_order;
            dummy_order.id = 999999; // Dummy identifier
            dummy_order.quantity = 10;
            dummy_order.is_buy = (rand() % 2 == 0);
            
            // Place dummy orders randomly around the spread
            int spread = (rand() % 3) + 1;
            dummy_order.price = dummy_order.is_buy ? 
                                (current_market_price - spread) : 
                                (current_market_price + spread);
            
            add_order(dummy_order);
        }

        // 4. BROADCAST NEW PRICE TO PYTHON
        redisReply *pub_reply = (redisReply*)redisCommand(c, "PUBLISH market_data %d", current_market_price);
        freeReplyObject(pub_reply);
    }

    redisFree(c);
    return 0;
}