/*
Let us understand this with the help of the following problem



Given two integers L and R return the sum of the segment [L, R]



The first step is constructing the segment tree with the addition operator and 0 as the neutral element.

If the range is one of the node’s range values then simply return the answer.
Otherwise, we will need to traverse the left and right children of the nodes and recursively continue the process till we find a node that covers a range that totally covers a part or whole of the range [L, R]
While returning from each call, we need to merge the answers received from each of its child.
Given an index,  idx, update the value of the array at index idx with value V



The element’s contribution is only in the path from its leaf to its parent. Thus only logN elements will get affected due to the update.

For updating, traverse till the leaf that stores the value of index idx and update the value. Then while tracing back in the path, modify the ranges accordingly.

The time complexity will be O(logN).
*/
#include <bits/stdc++.h>
using namespace std;
vector<int> A, ST;

void build(int node, int L, int R)
{

    // Leaf node where L == R
    if (L == R)
    {
        ST[node] = A[L];
    }
    else
    {

        // Find the middle element to
        // split the array into two halves
        int mid = (L + R) / 2;

        // Recursively travel the
        // left half
        build(2 * node, L, mid);

        // Recursively travel the
        // right half
        build(2 * node + 1, mid + 1, R);

        // Storing the sum of both the
        // children into the parent
        ST[node] = ST[2 * node] + ST[2 * node + 1];
    }
}

void update(int node, int L, int R, int idx, int val)
{

    // Find the lead node and
    // update its value
    if (L == R)
    {
        A[idx] += val;
        ST[node] += val;
    }
    else
    {

        // Find the mid
        int mid = (L + R) / 2;

        // If node value idx is at the
        // left part then update
        // the left part
        if (L <= idx and idx <= mid)
            update(2 * node, L, mid, idx, val);
        else
            update(2 * node + 1, mid + 1, R, idx, val);

        // Store the information in parents
        ST[node] = ST[2 * node] + ST[2 * node + 1];
    }
}

int query(int node, int tl, int tr, int l, int r)
{

    // If it lies out of range then
    // return 0
    if (r < tl or tr < l)
        return 0;

    // If the node contains the range then
    // return the node value
    if (l <= tl and tr <= r)
        return ST[node];
    int tm = (tl + tr) / 2;

    // Recursively traverse left and right
    // and find the node
    return query(2 * node, tl, tm, l, r) + query(2 * node + 1, tm + 1, tr, l, r);
}
// Driver code
int main()
{
    int n = 6;
    A = {0, 1, 3, 5, -2, 3};
    // Create a segment tree of size 4*n
    ST.resize(4 * n);
    // Build a segment tree
    build(1, 0, n - 1);
    cout << "Sum of values in range 0-4 are: "
         << query(1, 0, n - 1, 0, 4) << "\n";
    // Update the value at idx = 1 by
    // 100 thus becoming 101
    update(1, 0, n - 1, 1, 100);
    cout << "Value at index 1 increased by 100\n";
    cout << "sum of value in range 1-3 are: "
         << query(1, 0, n - 1, 1, 3) << "\n";
    return 0;
}