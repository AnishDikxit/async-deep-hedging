#include <iostream>
#include <vector>
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()
#include <thread>    // For sleep
#include <chrono>    // For milliseconds

using namespace std;

struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
};

vector<Order> bids; 
vector<Order> asks;

void add_order(Order new_order) {
    if (new_order.is_buy) {
        cout << "[Market] Incoming BUY  Order " << new_order.id << " willing to pay $" << new_order.price << "\n";
        
        for (int i = 0; i < asks.size(); i++) {
            if (asks[i].price <= new_order.price) {
                cout << "   -> TRADE EXECUTED! Buyer " << new_order.id 
                     << " bought from Seller " << asks[i].id 
                     << " at $" << asks[i].price << "\n\n";
                asks.erase(asks.begin() + i);
                return; 
            }
        }
        cout << "   -> No match found. Added to Bids.\n\n";
        bids.push_back(new_order); 
        
    } else {
        cout << "[Market] Incoming SELL Order " << new_order.id << " asking for $" << new_order.price << "\n";
        
        for (int i = 0; i < bids.size(); i++) {
            if (bids[i].price >= new_order.price) {
                cout << "   -> TRADE EXECUTED! Seller " << new_order.id 
                     << " sold to Buyer " << bids[i].id 
                     << " at $" << bids[i].price << "\n\n";
                bids.erase(bids.begin() + i);
                return; 
            }
        }
        cout << "   -> No match found. Added to Asks.\n\n";
        asks.push_back(new_order); 
    }
}

int main() {
    // 1. Initialize the Random Number Generator
    srand(time(0)); 
    
    int order_id_counter = 1;
    int current_market_price = 100; // We anchor the market around $100

    cout << "--- Starting Live Market Simulation ---\n";

    // 2. The Master Clock
    while (true) {
        Order new_order;
        new_order.id = order_id_counter++;
        new_order.quantity = 10;
        
        // 3. Flip a coin: 50% chance to be a Buy, 50% chance to be a Sell
        new_order.is_buy = (rand() % 2 == 0);
        
        // 4. Generate a random price close to $100 (between $98 and $102)
        // This ensures the buyers and sellers actually cross paths and trade!
        int price_fluctuation = (rand() % 5) - 2; 
        new_order.price = current_market_price + price_fluctuation;

        // 5. Send to your matching engine
        add_order(new_order);

        // 6. Pause for 500 milliseconds (half a second) so you can read it
        this_thread::sleep_for(chrono::milliseconds(500));
    }

    return 0;
}