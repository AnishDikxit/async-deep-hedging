#include <iostream>
#include <vector>
#include <queue>     // NEW: For priority_queue
#include <cstdlib>
#include <ctime>
#include <thread>
#include <chrono>

using namespace std;

struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
    long long execution_time; // NEW: When is this allowed to trade?
};

// NEW: This tells the priority_queue how to sort the orders.
// We want a "Min-Heap", meaning the LOWEST execution_time is at the top.
struct CompareOrder {
    bool operator()(Order const& o1, Order const& o2) {
        return o1.execution_time > o2.execution_time; 
    }
};

vector<Order> bids; 
vector<Order> asks;

// This function hasn't changed! It's exactly your logic from Phase 1.
void add_order(Order new_order) {
    if (new_order.is_buy) {
        for (int i = 0; i < asks.size(); i++) {
            if (asks[i].price <= new_order.price) {
                cout << "   -> TRADE EXECUTED! Buyer " << new_order.id 
                     << " bought from Seller " << asks[i].id 
                     << " at $" << asks[i].price << "\n\n";
                asks.erase(asks.begin() + i);
                return; 
            }
        }
        bids.push_back(new_order); 
    } else {
        for (int i = 0; i < bids.size(); i++) {
            if (bids[i].price >= new_order.price) {
                cout << "   -> TRADE EXECUTED! Seller " << new_order.id 
                     << " sold to Buyer " << bids[i].id 
                     << " at $" << bids[i].price << "\n\n";
                bids.erase(bids.begin() + i);
                return; 
            }
        }
        asks.push_back(new_order); 
    }
}

int main() {
    srand(time(0)); 
    
    // NEW: Our Time Machine (The Waiting Room)
    priority_queue<Order, vector<Order>, CompareOrder> latency_queue;

    long long current_time = 0; // Our master clock (simulating milliseconds)
    int order_id_counter = 1;
    int current_market_price = 100;

    cout << "--- Starting Latent Market Simulation ---\n";

    while (true) {
        // 1. Advance the master clock by 1 millisecond
        current_time++;

        // 2. Randomly generate an order (only 20% of the time, to space things out)
        if (rand() % 100 < 20) {
            Order new_order;
            new_order.id = order_id_counter++;
            new_order.quantity = 10;
            new_order.is_buy = (rand() % 2 == 0);
            new_order.price = current_market_price + ((rand() % 5) - 2);
            
            // THE CORE LATENCY LOGIC
            // We force this order to wait exactly 50 milliseconds before executing
            new_order.execution_time = current_time + 50; 
            
            latency_queue.push(new_order);
            
            string side = new_order.is_buy ? "BUY " : "SELL";
            cout << "[t=" << current_time << "] Received " << side << " Order " << new_order.id 
                 << " at $" << new_order.price 
                 << " -> Delayed until t=" << new_order.execution_time << "\n";
        }
        // 3. Check the Queue: Is it time to release an order?
        // We peek at the top order. If its time has come, we execute it!
        while (!latency_queue.empty() && latency_queue.top().execution_time <= current_time) {
            Order ready_order = latency_queue.top();
            latency_queue.pop();
            
            cout << "\n[t=" << current_time << "] RELEASING Order " << ready_order.id << " into the market!\n";
            add_order(ready_order); // NOW it finally hits your matching engine
        }

        // Slow down the terminal output so you can read it
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    return 0;
}