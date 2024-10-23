#include <iostream>
#include <cmath>
#include <algorithm>
#include <bits/stdc++.h>
#define yes cout << "YES\n"
#define no cout << "NO\n"
using namespace std;
#define ll long long int
typedef vector<ll> vll;
typedef deque<ll> dqll;
typedef vector<vector<ll>> vvll;
typedef map<ll, ll> mpll;
typedef map<char, ll> mpcll;
typedef priority_queue<ll, vector<ll>, greater<ll>> min_heap;
typedef priority_queue<ll> max_heap;
bool descendingOrder(ll a, ll b)
{
    return a >= b;
}
bool isinteger(double number)
{
    double rounded = floor(number);
    return rounded == number;
}
bool isSquare(ll num)
{
    double root = sqrt(static_cast<double>(num));
    return root == floor(root);
}
vector<ll> primeFactors(ll num)
{
    vector<ll> factors;

    // Print the number of 2s that divide num
    while (num % 2 == 0)
    {
        factors.push_back(2);
        num = num / 2;
    }

    // num must be odd at this point. So we can skip one element (Note i = i +2)
    for (ll i = 3; i <= sqrt(num); i = i + 2)
    {
        // While i divides num, print i and divide num
        while (num % i == 0)
        {
            factors.push_back(i);
            num = num / i;
        }
    }

    // This condition is to handle the case when num is a prime number greater than 2
    if (num > 2)
        factors.push_back(num);

    return factors;
}
ll findSubarraySum(vector<ll> arr, ll n, ll sum)
{
    unordered_map<ll, ll> prevSum;

    ll res = 0;
    ll currSum = 0;

    for (ll i = 0; i < n; i++)
    {

        currSum += arr[i];

        if (currSum == sum)
            res++;

        if (prevSum.find(currSum - sum) != prevSum.end())
            res += (prevSum[currSum - sum]);

        prevSum[currSum]++;
    }

    return res;
}
ll lcm(ll a, ll b)
{
    return (a * b) / (__gcd(a, b));
}
vector<ll> LongestNonDecreasingSubsequence(vector<ll> &v, ll n)
{
    if (v.size() == 0)
        return {};

    vector<ll> tailIndex(n, 0);
    vector<ll> prevIndex(n, -1);
    ll length = 1;

    tailIndex[0] = 0;

    for (ll i = 1; i < v.size(); i++)
    {
        auto b = tailIndex.begin(), e = tailIndex.begin() + length;
        auto it = upper_bound(b, e, i, [&v](ll idx1, ll idx2)
                              { return v[idx1] < v[idx2]; });
        if (it == tailIndex.begin() + length)
        {
            tailIndex[length] = i;
            prevIndex[i] = tailIndex[length - 1];
            length++;
        }
        else
        {
            *it = i;
            prevIndex[i] = (it == tailIndex.begin()) ? -1 : *(it - 1);
        }
    }
    vector<ll> result;
    if (length > 0)
    {
        ll currentIndex = tailIndex[length - 1];
        while (currentIndex != -1)
        {
            result.push_back(currentIndex);
            currentIndex = prevIndex[currentIndex];
        }
        reverse(result.begin(), result.end());
    }

    return result;
}
std::vector<long long> getDivisors(long long n)
{
    std::vector<long long> divisors;
    for (long long i = 1; i * i <= n; i++)
    {
        if (n % i == 0)
        {
            divisors.push_back(i);
            if (n / i != i)
            {
                divisors.push_back(n / i);
            }
        }
    }
    return divisors;
}
map<char, vector<ll>> calculateSuffixFrequencies(const string &str)
{
    ll n = str.size();
    map<char, vector<ll>> freq;
    for (char c = 'a'; c <= 'z'; ++c)
    {
        freq[c] = vector<ll>(n, 0);
    }

    freq[str[n - 1]][n - 1]++;
    for (ll i = n - 2; i >= 0; --i)
    {
        for (char c = 'a'; c <= 'z'; ++c)
        {
            freq[c][i] = freq[c][i + 1];
        }
        freq[str[i]][i]++;
    }
    return freq;
}
map<char, vector<ll>> calculateOccurrencePositions(const string &str)
{
    map<char, vector<ll>> occurrencePositions;

    for (ll i = 0; i < str.size(); ++i)
    {
        occurrencePositions[str[i]].push_back(i);
    }

    return occurrencePositions;
}
ll maxElements(std::vector<ll> &arr, ll M)
{
    std::sort(arr.begin(), arr.end());
    ll left = 0, right = 0;
    long long sum = 0;
    ll maxLen = 0;
    while (right < arr.size())
    {
        if (right > 0 && arr[right] != arr[right - 1])
        {
            long long diff = arr[right] - arr[right - 1];
            if (diff > M)
            {
                left = right;
                sum = 0;
            }
            else
            {
                sum += diff * (right - left);
            }
        }
        while (sum > M)
        {
            sum -= arr[right] - arr[left];
            left++;
        }
        maxLen = std::max(maxLen, right - left + 1);
        right++;
    }
    return maxLen;
}
ll maxSubArraySum(vector<ll> a, ll constnt)
{
    ll size = a.size();
    vector<ll> dp(size, 0);
    dp[0] = a[0];
    ll ans = dp[0];
    for (ll i = 1; i < size; i++)
    {
        dp[i] = max(a[i], (a[i] + dp[i - 1]));
        ans = max(ans, dp[i]);
    }
    // cout << "d " << ans << endl;
    return ans;
}
vector<ll> findDivisors(ll n)
{
    vector<ll> divisors;
    for (ll i = 1; i <= sqrt(n); i++)
    {
        if (n % i == 0)
        {
            // If divisors are not equal, push both
            if (n / i == i)
            {
                divisors.push_back(i);
            }
            else
            {
                divisors.push_back(i);
                divisors.push_back(n / i);
            }
        }
    }
    return divisors;
}
void modifyVector(const std::vector<ll> &vec1, std::vector<ll> &vec2) // vec1 is the original vector and vec2 is the modified vector...suppose 1,-2-2,3,3=1,-4,6
{
    ll sum = vec1[0];
    for (size_t i = 1; i < vec1.size(); ++i)
    {
        if ((vec1[i] > 0 && vec1[i - 1] > 0) || (vec1[i] < 0 && vec1[i - 1] < 0))
        {
            sum += vec1[i];
        }
        else
        {
            vec2.push_back(sum);

            // cout << sum << "s" << endl;
            sum = vec1[i];
        }
    }
    vec2.push_back(sum);
    // cout << vec2.size() << endl;
    // cout << sum << "s" << endl;
}
void bfs(vector<vector<int>> &adj, int s)
{
    // Create a queue for BFS
    queue<int> q;

    // Initially mark all the vertices as not visited
    // When we push a vertex into the q, we mark it as
    // visited
    vector<bool> visited(adj.size(), false);

    // Mark the source node as visited and
    // enqueue it
    visited[s] = true;
    q.push(s);

    // Iterate over the queue
    while (!q.empty())
    {

        // Dequeue a vertex from queue and print it
        int curr = q.front();
        q.pop();
        cout << curr << " ";

        // Get all adjacent vertices of the dequeued
        // vertex curr If an adjacent has not been
        // visited, mark it visited and enqueue it
        for (int x : adj[curr])
        {
            if (!visited[x])
            {
                visited[x] = true;
                q.push(x);
            }
        }
    }
}
/*dfs.............
void dfsUtil(int v, vector<vector<int>>& adj, vector<bool>& visited) {
    // Mark the current node as visited and print it
    visited[v] = true;
    cout << v << " ";

    // Recur for all the vertices adjacent to this vertex
    for (int x : adj[v]) {
        if (!visited[x]) {
            dfsUtil(x, adj, visited);
        }
    }
}

void dfs(vector<vector<int>>& adj, int s) {
    // Initially mark all the vertices as not visited
    vector<bool> visited(adj.size(), false);

    // Call the recursive helper function to print DFS traversal
    dfsUtil(s, adj, visited);
}*/
bool check_prime(long long int n)
{
    // Edge cases: numbers less than 2 are not prime
    if (n <= 1)
        return false;
    if (n <= 3)
        return true; // 2 and 3 are prime

    // Eliminate even numbers and multiples of 3
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    // Check for factors from 5 to sqrt(n)
    for (long long int i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }

    return true;
}
bool checkLinearCombination(std::vector<long long int> &array, long long int value)
{
    for (size_t i = 0; i < array.size() - 1; i++)
    { // Use size_t for indexing
        for (size_t j = i + 1; j < array.size(); j++)
        {
            if (value % std::__gcd(array[i], array[j]) == 0)
            {
                printf("x=%lld, y=%lld, i=%zu, j=%zu\n", array[0], array[1], i, j);
                return true;
            }
        }
    }
    return false;
}

int maxSubsequenceSubstring(const string &a, const string &b)
{
    // substring of b as subsequence in a
    int m = a.size(), n = b.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (a[i] == b[j])
            {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            }
            else
            {
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }

    int maxLength = 0;
    // Find the maximum in the last row of dp
    for (int j = 0; j <= n; ++j)
    {
        maxLength = max(maxLength, dp[m][j]);
    }

    return maxLength;
}
bool checkallequal(vector<ll> &vec)
{
    for (int i = 1; i < vec.size(); i++)
    {
        if (vec[i] != vec[i - 1])
        {
            return false;
        }
    }
    return true;
}
ll gettotalNumOfInversions(vector<ll> &A)
{
    ll N = A.size();
    if (N <= 1)
    {
        return 0;
    }

    priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>> sortList;
    ll result = 0;

    // Heapsort, O(N*log(N))
    for (ll i = 0; i < N; i++)
    {
        sortList.push(make_pair(A[i], i));
    }

    // Create a sorted list of indexes
    vector<ll> x;
    while (!sortList.empty())
    {

        // O(log(N))
        ll v = sortList.top().first;
        ll i = sortList.top().second;
        sortList.pop();

        // Find the current minimum's index
        // the index y can represent how many minimums on the left
        ll y = upper_bound(x.begin(), x.end(), i) - x.begin();

        // i can represent how many elements on the left
        // i - y can find how many bigger nums on the left
        result += i - y;

        x.insert(upper_bound(x.begin(), x.end(), i), i);
    }

    return result;
}
vector<ll> getInversionCountsForEachElement(vector<ll> &A)
{
    ll N = A.size();
    vector<ll> inversionCounts(N, 0); // Initialize inversion counts for each element

    if (N <= 1)
    {
        return inversionCounts; // If the array is of size 1 or empty, return the counts as is
    }

    priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>> sortList;

    // Heapsort, O(N*log(N))
    for (ll i = 0; i < N; i++)
    {
        sortList.push(make_pair(A[i], i));
    }

    vector<ll> x; // Sorted list of indexes
    while (!sortList.empty())
    {
        ll v = sortList.top().first;
        ll i = sortList.top().second;
        sortList.pop();

        ll y = upper_bound(x.begin(), x.end(), i) - x.begin();

        // Instead of adding to a total result, update the inversion count for the specific element
        inversionCounts[i] = i - y;

        x.insert(upper_bound(x.begin(), x.end(), i), i);
    }

    return inversionCounts;
}
vector<vector<ll>> computebitwisePrefixSums(const vector<ll> &nums)
{
    // Assuming 32 bits for each number
    ll bits = 32;
    vector<vector<ll>> prefixSums(bits, vector<ll>(nums.size() + 1, 0));

    for (ll bit = 0; bit < bits; ++bit)
    {
        for (size_t i = 0; i < nums.size(); ++i)
        {
            // If the current bit is set in nums[i], add 1 to the current prefix sum
            prefixSums[bit][i + 1] = prefixSums[bit][i] + ((nums[i] >> bit) & 1);
        }
    }

    return prefixSums;
}
ll bitwiseAndSubsegment(const vector<vector<ll>> &prefixSums, ll l, ll r)
{
    int result = 0;
    int segmentLength = r - l + 1;

    for (ll bit = 0; bit < 32; ++bit)
    {
        // Check if all values in the interval have this bit set
        if (prefixSums[bit][r + 1] - prefixSums[bit][l] == segmentLength)
        {
            // Set this bit in the result
            result |= (1 << bit);
        }
    }

    return result;
}
ll bitwiseXorSubsegment(const vector<vector<ll>> &prefixSums, ll l, ll r)
{
    ll result = 0;
    int segmentLength = r - l + 1;

    for (ll bit = 0; bit < 32; ++bit)
    {
        // Calculate the number of set bits in the interval for the current bit position
        ll setBits = prefixSums[bit][r + 1] - prefixSums[bit][l];
        // If the number of set bits is odd, the result for this bit position is 1
        if (setBits % 2 == 1)
        {
            result |= (1LL << bit);
        }
    }

    return result;
}
ll bitwiseOrSubsegment(const vector<vector<ll>> &prefixSums, ll l, ll r)
{
    ll result = 0;

    for (ll bit = 0; bit < 32; ++bit)
    {
        // If there's at least one set bit in the interval for the current bit position, set this bit in the result
        if (prefixSums[bit][r + 1] - prefixSums[bit][l] > 0)
        {
            result |= (1LL << bit);
        }
    }

    return result;
}
vector<ll> binaryrepresentation(ll n)
{
    vector<ll> binary;
    while (n > 0)
    {
        binary.push_back(n % 2);
        n /= 2;
    }
    reverse(binary.begin(), binary.end());
    return binary;
}
vector<ll> binary62representation(ll n)
{
    vector<ll> binary(62, 0); // Initialize a vector with 62 elements set to 0
    int index = 61;           // Start from the least significant bit

    while (n > 0 && index >= 0)
    {
        binary[index] = n % 2;
        n /= 2;
        index--;
    }

    return binary;
}
ll binarytonumber(vector<ll> binary)
{
    ll number = 0;
    for (ll i = 0; i < binary.size(); i++)
    {
        number += binary[i] * pow(2, binary.size() - i - 1);
    }
    return number;
}
bool balancedbrackets(string s)
{
    stack<char> st;
    for (ll i = 0; i < s.size(); i++)
    {
        if (s[i] == '(' || s[i] == '{' || s[i] == '[')
        {
            st.push(s[i]);
        }
        else
        {
            if (st.empty())
            {
                return false;
            }
            if (s[i] == ')' && st.top() == '(')
            {
                st.pop();
            }
            else if (s[i] == '}' && st.top() == '{')
            {
                st.pop();
            }
            else if (s[i] == ']' && st.top() == '[')
            {
                st.pop();
            }
            else
            {
                return false;
            }
        }
    }
    return st.empty();
}
ll calculate_cost_of_regularbracketsequence(const string &s, ll n)
{
    //((())) cost=4-3+5-2+6-1=9
    stack<ll> positions;
    ll cost = 0;

    for (ll i = 0; i < n; ++i)
    {
        if (s[i] == '(')
        {
            positions.push(i);
        }
        else if (s[i] == ')')
        {
            if (!positions.empty())
            {
                int open_pos = positions.top();
                positions.pop();
                cost += (i - open_pos);
            }
        }
    }

    return cost;
}
ll nCr(ll n, ll r)
{
    // If r is greater than n, return 0
    if (r > n)
        return 0;
    // If r is 0 or equal to n, return 1
    if (r == 0 || n == r)
        return 1;
    // Initialize the logarithmic sum to 0
    double res = 0;
    // Calculate the logarithmic sum of the numerator and denominator using loop
    for (ll i = 0; i < r; i++)
    {
        // Add the logarithm of (n-i) and subtract the logarithm of (i+1)
        res += log(n - i) - log(i + 1);
    }
    // Convert logarithmic sum back to a normal number
    return (ll)round(exp(res));
}
// xor observation: if X is bitwise xor of some elements and we want to remove any eleemnt y from it, then  X=X^y..as y^y=0
ll segmentsum(map<ll, ll> &prefixsum, ll l, ll r)
{
    if (l > r)
    {
        return 0;
    }
    return prefixsum[r] - prefixsum[l - 1];
}
const ll MOD = 1000000007;

ll factorial_mod(ll n)
{
    ll result = 1;
    for (ll i = 1; i <= n; ++i)
    {
        result = (result * i) % MOD;
    }
    return result;
}
ll find_mex(const set<ll> &s)
{
    ll mex_value = 0;
    while (s.find(mex_value) != s.end())
    {
        mex_value++;
    }
    return mex_value;
    // for finding second mex insert mex_value in set and again find mex
}
ll find_mex_sortedvector(vector<ll> &vec)
{
    ll ans = 0;
    for (ll i = 0; i < vec.size(); i++)
    {
        if (vec[i] == ans)
        {
            ans++;
        }
        else
        {
            break;
        }
    }
    return ans;
}
ll countOverlappingIntervals(const vector<pair<ll, ll>> &vec, ll startday, ll endday)
{
    auto startIt = lower_bound(vec.begin(), vec.end(), make_pair(startday, LLONG_MIN));
    auto endIt = upper_bound(vec.begin(), vec.end(), make_pair(endday, LLONG_MAX));
    return distance(startIt, endIt);
}
ll countSetBits(ll n)
{
    ll count = 0;
    while (n)
    {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
std::string longestSubstringInString(const std::string &s1, const std::string &s2)
{
    std::string longestSubstr;
    size_t maxLength = 0;

    for (size_t i = 0; i < s1.length(); ++i)
    {
        for (size_t j = i + 1; j <= s1.length(); ++j)
        {
            std::string substr = s1.substr(i, j - i);
            if (s2.find(substr) != std::string::npos && substr.length() > maxLength)
            {
                longestSubstr = substr;
                maxLength = substr.length();
            }
        }
    }

    return longestSubstr;
}
std::string longestCommonSubsequence(const std::string &s1, const std::string &s2)
{
    int m = s1.size();
    int n = s2.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    // Fill the dp array
    for (int i = 1; i <= m; ++i)
    {
        for (int j = 1; j <= n; ++j)
        {
            if (s1[i - 1] == s2[j - 1])
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            else
            {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // Reconstruct the LCS
    std::string lcs;
    int i = m, j = n;
    while (i > 0 && j > 0)
    {
        if (s1[i - 1] == s2[j - 1])
        {
            lcs.push_back(s1[i - 1]);
            --i;
            --j;
        }
        else if (dp[i - 1][j] > dp[i][j - 1])
        {
            --i;
        }
        else
        {
            --j;
        }
    }

    // The lcs string is constructed backwards, so reverse it
    std::reverse(lcs.begin(), lcs.end());
    return lcs;
}
std::string common_subsequence(const std::string &str1, const std::string &str2)
{
    int len1 = str1.size(), len2 = str2.size();
    std::vector<std::vector<int>> dp_table(len1 + 1, std::vector<int>(len2 + 1, 0));
    for (int x = 0; x < len1; ++x)
    {
        for (int y = 0; y < len2; ++y)
        {
            if (str1[x] == str2[y])
                dp_table[x + 1][y + 1] = dp_table[x][y] + 1;
            else
                dp_table[x + 1][y + 1] = std::max(dp_table[x + 1][y], dp_table[x][y + 1]);
        }
    }
    int x = len1, y = len2;
    std::string result;

    while (x > 0 && y > 0)
    {
        if (str1[x - 1] == str2[y - 1])
        {
            result.push_back(str1[x - 1]);
            --x;
            --y;
        }
        else if (dp_table[x - 1][y] >= dp_table[x][y - 1])
            --x;
        else
            --y;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

bool isSubstring(const std::string &str1, const std::string &str2)
{
    return str2.find(str1) != std::string::npos;
}
long long power2mod(ll n)
{
    ll r = 1;
    ll b = 2;

    while (n > 0)
    {
        if (n % 2 == 1)
        {
            r = (r * b) % MOD;
        }
        b = (b * b) % MOD;
        n /= 2;
    }

    return r;
}
// idea to be noted: sometimes can generate resources before test cases start and use them in test cases wo that runtime does not added extra
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    ll t;
    cin >> t;
    while (t--)
    {
    }

    return 0;
}