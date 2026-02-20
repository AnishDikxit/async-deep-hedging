#include <bits/stdc++.h>
using namespace std;
struct Order{
    int id;
    int price;
    int quantity;
    bool is_buy;
};
vector<int> bids, asks;
void add_order(Order new_order){
    if(new_order.is_buy){
        for(int i = 0; i<asks.size(); i++){
            if(asks[i]<=100){
                asks.erase(asks.begin()+i);
                cout<<"Trade Executed!\n";
                return;
            }
        }
        bids.push_back(new_order.price);
    }
    else{
        asks.push_back(new_order.price);
    }
    return;
}
int main(){
    
    Order order1, order2;
    order1.price = 100;
    order1.is_buy = false;
    add_order(order1);
    order2.price = 100;
    order2.is_buy = true;
    add_order(order2);
    return 0;
}