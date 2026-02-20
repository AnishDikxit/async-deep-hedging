#include <iostream>
#include <vector>

using namespace std;

struct Order {
    int id;
    int price;
    int quantity;
    bool is_buy;
};

// FIX 3: Store the actual Order objects, not just the integers
vector<Order> bids; 
vector<Order> asks;

void add_order(Order new_order) {
    if (new_order.is_buy) {
        // --- A BUYER ENTERS THE MARKET ---
        for (int i = 0; i < asks.size(); i++) {
            // FIX 1: Compare against the incoming order's price
            if (asks[i].price <= new_order.price) {
                cout << "Trade Executed! Buyer " << new_order.id 
                     << " bought from Seller " << asks[i].id 
                     << " at price " << asks[i].price << "\n";
                
                asks.erase(asks.begin() + i);
                return; // Trade complete, exit the function
            }
        }
        // No seller found, add to the waiting list
        bids.push_back(new_order); 
        
    } else {
        // --- A SELLER ENTERS THE MARKET ---
        // FIX 2: Check if there are any waiting buyers willing to pay this price!
        for (int i = 0; i < bids.size(); i++) {
            if (bids[i].price >= new_order.price) {
                cout << "Trade Executed! Seller " << new_order.id 
                     << " sold to Buyer " << bids[i].id 
                     << " at price " << bids[i].price << "\n";
                
                bids.erase(bids.begin() + i);
                return; // Trade complete, exit the function
            }
        }
        // No buyer found, add to the waiting list
        asks.push_back(new_order); 
    }
}

int main() {
    // 1. Seller enters, wants $100
    Order order1 = {1, 100, 10, false}; 
    add_order(order1);
    
    // 2. Buyer enters, willing to pay up to $105
    Order order2 = {2, 105, 10, true};  
    add_order(order2);
    
    return 0;
}