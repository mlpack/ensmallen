#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// define LDP struct
struct LDP {
    // define coordinates of the lower and upper points of the LDP
    vector<double> lower;
    vector<double> upper;
};

// define PS struct
struct PS {
    // define vertices of the PS
    vector<vector<double>> vertices;
};

// define weight struct
struct Weight {
    // define coordinates of the weight vector
    vector<double> coordinates;
};

// define Facet struct
struct Facet {
    // define vertices of the facet
    vector<vector<double>> vertices;
    // define normal vector of the facet
    vector<double> normal;
    // define LDP associated with the facet
    LDP ldp;
};

// define function to calculate the LDP for a given weight vector
LDP calculate_ldp(PS ps, Weight w);

// define function to compute the set of facets of a given PS
vector<Facet> compute_facets(PS ps);

// define function to remove facets from the set based on their associated LDPs
void remove_facets(vector<Facet>& facets, vector<LDP> ldps, double stop_tolerance);

// define function to calculate the distance between a facet and its associated LDP
double calculate_distance(Facet facet);

// define function to choose the next facet to run
Facet choose_next_facet(vector<Facet> facets, double stop_tolerance, bool boundary_option);

// define function to compute the next weight vector and run it
Weight compute_next_weight(Facet facet);

int main() {
    // main code here
    return 0;
}
