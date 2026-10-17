#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <cstdint>
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
    // Audit M1: the book holds only random orders. It does not change the price or the agent fills.
    // The model thus has no market impact, no liquidity limit and no variable spread.
    std::vector<Order> bids; 
    std::vector<Order> asks;
    std::priority_queue<Order, std::vector<Order>, CompareOrder> latency_queue;
    
    long long current_time; 
    int order_id_counter;
    int current_market_price;
    double hawkes_excitation;

    // Audit B1/B6: one seeded generator for each simulator.
    // rand() is shared by the process and srand(time(0)) gives the same seed to workers that start in the same second.
    std::mt19937 rng;
    std::uniform_real_distribution<double> unit_dist{0.0, 1.0};

    int rand_int(int low, int high) { // inclusive range, no modulo bias
        return std::uniform_int_distribution<int>(low, high)(rng);
    }

    const double base_mu = 0.02;           
    // Audit C2: alpha_jump was 0.15. The code decays first and then adds alpha, so the
    // branching ratio is n = alpha * decay / (1 - decay). With 0.15, n = 7.35 (unstable).
    // The default 0.012 gives n = 0.588. The stable event rate is then base_mu / (1 - n) = 0.0485 per tick.
    double alpha_jump;
    const double decay_rate = 0.98;        
    long long event_count;

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
    // Audit B1: the caller gives the seed. A fixed seed makes a run reproducible.
    explicit MarketSimulator(std::uint32_t seed = 0, double alpha_jump = 0.012)
        : rng(seed), alpha_jump(alpha_jump) {
        reset();
    }

    // A negative seed keeps the current generator state. A seed >= 0 restarts it.
    void reset(std::int64_t seed = -1) {
        if (seed >= 0) rng.seed(static_cast<std::uint32_t>(seed));
        bids.clear();
        asks.clear();
        latency_queue = std::priority_queue<Order, std::vector<Order>, CompareOrder>();
        current_time = 0;
        order_id_counter = 1;
        current_market_price = 100;
        hawkes_excitation = 0.0;
        event_count = 0;
    }

    long long get_event_count() const { return event_count; }
    double get_branching_ratio() const { return alpha_jump * decay_rate / (1.0 - decay_rate); }

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
        // The intensity is used as a probability for each tick, so it must not exceed 1.
        current_lambda = std::min(current_lambda, 1.0);

        if (unit_dist(rng) < current_lambda) {
            hawkes_excitation += alpha_jump;
            event_count++;
            current_market_price += rand_int(-2, 2); 
            if (current_market_price < 1) current_market_price = 1;
            
            int burst_orders = rand_int(2, 5); 
            for (int i = 0; i < burst_orders; i++) {
                Order dummy_order;
                dummy_order.id = 999999; 
                dummy_order.quantity = 10;
                dummy_order.is_buy = (rand_int(0, 1) == 0);
                int spread = rand_int(1, 3);
                dummy_order.price = dummy_order.is_buy ? (current_market_price - spread) : (current_market_price + spread);
                dummy_order.expiration_time = current_time + 50 + rand_int(0, 249);
                add_order(dummy_order);
            }
        }

        std::vector<FillReport> executed_fills;

        // 2. Process Latency Queue (Slippage happens here!)
        while (!latency_queue.empty() && latency_queue.top().execution_time <= current_time) {
            Order ready_order = latency_queue.top();
            latency_queue.pop();
            
            // If it is the AI's order (not a dummy order), it executes at the CURRENT mutated price
            // Audit C6: the order price is ignored. Agent orders are market orders with a full fill.
            if (ready_order.id != 999999) {
                FillReport report;
                report.order_id = ready_order.id;
                report.execution_price = current_market_price; // The price 50ms LATER
                report.quantity = ready_order.quantity;
                report.is_buy = ready_order.is_buy;
                executed_fills.push_back(report);
            } else {
                // Audit M1: this branch never runs. Random orders go to the book directly, not to the queue.
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
        .def(py::init<std::uint32_t, double>(), py::arg("seed") = 0, py::arg("alpha_jump") = 0.012)
        .def("reset", &MarketSimulator::reset, py::arg("seed") = -1)
        .def("place_order", &MarketSimulator::place_order)
        .def("step", &MarketSimulator::step)
        .def("get_event_count", &MarketSimulator::get_event_count)
        .def("get_branching_ratio", &MarketSimulator::get_branching_ratio);
}