#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // CRITICAL: Allows returning vectors and pairs to Python

namespace py = pybind11;

// --- DATA STRUCTURES ---
struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
    long long execution_time; 
    long long expiration_time;
};

// NEW: The exact receipt of execution sent back to Python
struct FillReport {
    int order_id;
    int execution_price;
    int quantity;
    bool is_buy;
};

struct CompareOrder {
    bool operator()(Order const& o1, Order const& o2) {
        return o1.execution_time > o2.execution_time; 
    }
};

class MarketSimulator {
private:
    std::vector<Order> bids; 
    std::vector<Order> asks;
    std::priority_queue<Order, std::vector<Order>, CompareOrder> latency_queue;
    
    long long current_time; 
    int order_id_counter;
    int current_market_price;
    double hawkes_excitation;

    const double base_mu = 0.02;           
    const double alpha_jump = 0.15;        
    const double decay_rate = 0.98;        

    void add_order(Order new_order) {
        if (new_order.is_buy) {
            for (size_t i = 0; i < asks.size(); i++) {
                if (asks[i].price <= new_order.price) {
                    if(new_order.id != 999999 && asks[i].id != 999999) continue;
                    asks.erase(asks.begin() + i);
                    return; 
                }
            }
            bids.push_back(new_order); 
        } else {
            for (size_t i = 0; i < bids.size(); i++) {
                if (bids[i].price >= new_order.price) {
                    if(new_order.id != 999999 && bids[i].id != 999999) continue;
                    bids.erase(bids.begin() + i);
                    return; 
                }
            }
            asks.push_back(new_order); 
        }
    }

    void cleanup_expired_orders() {
        bids.erase(std::remove_if(bids.begin(), bids.end(),
            [this](const Order& o) { return o.expiration_time != 0 && o.expiration_time <= current_time; }), bids.end());
        asks.erase(std::remove_if(asks.begin(), asks.end(),
            [this](const Order& o) { return o.expiration_time != 0 && o.expiration_time <= current_time; }), asks.end());
    }

public:
    MarketSimulator() {
        srand(time(0));
        reset();
    }

    void reset() {
        bids.clear();
        asks.clear();
        latency_queue = std::priority_queue<Order, std::vector<Order>, CompareOrder>();
        current_time = 0;
        order_id_counter = 1;
        current_market_price = 100;
        hawkes_excitation = 0.0;
    }

    void place_order(int price, bool is_buy) {
        Order new_order;
        new_order.id = order_id_counter++;
        new_order.quantity = 10;
        new_order.price = price;
        new_order.is_buy = is_buy;
        new_order.execution_time = current_time + 50; // 50ms simulated latency
        new_order.expiration_time = 0; 
        latency_queue.push(new_order);
    }

    // NEW: Returns a Pair -> (New Market Price, List of Executed Orders)
    std::pair<int, std::vector<FillReport>> step() {
        current_time++;

        // 1. Hawkes Process (Market Mutates)
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
        }

        std::vector<FillReport> executed_fills;

        // 2. Process Latency Queue (Slippage happens here!)
        while (!latency_queue.empty() && latency_queue.top().execution_time <= current_time) {
            Order ready_order = latency_queue.top();
            latency_queue.pop();
            
            // If it is the AI's order (not a dummy order), it executes at the CURRENT mutated price
            if (ready_order.id != 999999) {
                FillReport report;
                report.order_id = ready_order.id;
                report.execution_price = current_market_price; // The price 50ms LATER
                report.quantity = ready_order.quantity;
                report.is_buy = ready_order.is_buy;
                executed_fills.push_back(report);
            } else {
                add_order(ready_order); 
            }
        }

        // 3. Sweep
        if(current_time % 50 == 0) cleanup_expired_orders();

        return std::make_pair(current_market_price, executed_fills);
    }
};

// --- PYBIND11 MODULE BINDINGS ---
PYBIND11_MODULE(cpp_exchange, m) {
    m.doc() = "Asynchronous C++ Market Simulator for PyTorch";

    // Expose the FillReport struct so Python can read its attributes
    py::class_<FillReport>(m, "FillReport")
        .def_readonly("order_id", &FillReport::order_id)
        .def_readonly("execution_price", &FillReport::execution_price)
        .def_readonly("quantity", &FillReport::quantity)
        .def_readonly("is_buy", &FillReport::is_buy);

    py::class_<MarketSimulator>(m, "MarketSimulator")
        .def(py::init<>())
        .def("reset", &MarketSimulator::reset)
        .def("place_order", &MarketSimulator::place_order)
        .def("step", &MarketSimulator::step);
}