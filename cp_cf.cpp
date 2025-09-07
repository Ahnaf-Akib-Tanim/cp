#include <bits/stdc++.h>
#define yes cout << "YES\n"
#define no cout << "NO\n"
using namespace std;
#define ll long long int
typedef vector<ll> vll;
typedef deque<ll> dqll;
typedef vector<vector<ll>> vvll;
typedef vector<bool> vb;
typedef map<ll, ll> mpll;
typedef map<char, ll> mpcll;
typedef unordered_map<ll, ll> umpll;
typedef priority_queue<ll, vector<ll>, greater<ll>> min_heap;
typedef priority_queue<ll> max_heap;
typedef pair<ll, ll> pll;
typedef vector<pll> vpll;
#define takein1(n, v)           \
    for (ll i = 1; i <= n; i++) \
    {                           \
        cin >> v[i];            \
    }
#define takein(n, v)           \
    for (ll i = 0; i < n; i++) \
    {                          \
        cin >> v[i];           \
    }
#define calcfreq(mpll, v)             \
    for (ll i = 0; i < v.size(); i++) \
    {                                 \
        mpll[v[i]]++;                 \
    }
#define printv(v)         \
    for (auto x : v)      \
    {                     \
        cout << x << " "; \
    }                     \
    cout << endl;
#define printvv(v)            \
    for (auto x : v)          \
    {                         \
        for (auto y : x)      \
        {                     \
            cout << y << " "; \
        }                     \
        cout << endl;         \
    }
struct pair_hash
{
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};
#define printa(a) \
    cout << a << endl;
#define maxi(a, b, c) max(a, max(b, c))
#define sortv(v) sort(v.begin(), v.end());
#define sortrv(v) sort(v.begin(), v.end(), greater<ll>());
// customized heap
//  ll distance(const pair<ll, ll>& p) {
//      if (p.first % 3 == 2 && p.second % 3 == 2)
//          return p.first + p.second + 2;
//      return p.first + p.second;
//  }

// // Custom comparator for the min-heap.
// // Note: We invert the logic because std::priority_queue is a max-heap by default.
// struct CustomCmp {
//     bool operator()(const pair<ll, ll>& a, const pair<ll, ll>& b) const {
//         ll da = distance(a), db = distance(b);
//         if (da == db)
//             return a.first > b.first;  // invert: lower first gets higher priority.
//         return da > db;  // invert: lower distance gets higher priority.
//     }
// };
// priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, CustomCmp> minHeap;
typedef unordered_map<pair<ll, ll>, bool, pair_hash> pairexists;
struct Compare
{
    bool operator()(const pair<ll, ll> &a, const pair<ll, ll> &b)
    {
        return a.second > b.second;
    }
};
struct Compare2
{
    bool operator()(const pair<ll, ll> &a, const pair<ll, ll> &b)
    {
        return a.second < b.second;
    }
};
typedef priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, Compare> min_heap_key_val;
typedef priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, Compare2> max_heap_key_val;
struct Line
{
    ll A, B, C;
};
Line normalize_line(ll x1, ll y1, ll x2, ll y2)
{
    ll A = y2 - y1;
    ll B = x1 - x2;
    ll C = x2 * y1 - x1 * y2;
    if (A < 0 || (A == 0 && B < 0))
    {
        A = -A;
        B = -B;
        C = -C;
    }
    ll g = __gcd(abs(A), __gcd(abs(B), abs(C)));
    if (g != 0)
    {
        A /= g;
        B /= g;
        C /= g;
    }

    return {A, B, C};
}
struct LineHash
{
    size_t operator()(const Line &l) const
    {
        auto h1 = std::hash<ll>()(l.A);
        auto h2 = std::hash<ll>()(l.B);
        auto h3 = std::hash<ll>()(l.C);
        return ((h1 * 31 + h2) * 31 + h3);
    }
};

struct LineEqual
{
    bool operator()(const Line &l1, const Line &l2) const
    {
        return l1.A == l2.A && l1.B == l2.B && l1.C == l2.C;
    }
};
// unordered_set<Line, LineHash, LineEqual> lines;
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
    while (num % 2 == 0)
    {
        factors.push_back(2);
        num = num / 2;
    }
    for (ll i = 3; i <= sqrt(num); i = i + 2)
    {
        while (num % i == 0)
        {
            factors.push_back(i);
            num = num / i;
        }
    }
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
    queue<int> q;
    vector<bool> visited(adj.size(), false);
    visited[s] = true;
    q.push(s);
    while (!q.empty())
    {
        int curr = q.front();
        q.pop();
        cout << curr << " ";
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
{ //((())) cost=4-3+5-2+6-1=9
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
vvll allcombinations(vll &vec)
{
    vvll result;
    ll n = vec.size();
    for (ll i = 0; i < (1 << n); i++)
    {
        vll subset;
        for (ll j = 0; j < n; j++)
        {
            if (i & (1 << j))
            {
                subset.push_back(vec[j]);
            }
        }
        result.push_back(subset);
    }
    return result;
}
struct DSU
{
    vll e;
    DSU(ll N) { e = vector<ll>(N, -1); }

    // get representive component (uses path compression)
    ll get(ll x) { return e[x] < 0 ? x : e[x] = get(e[x]); }

    bool same_set(ll a, ll b) { return get(a) == get(b); }

    ll size(ll x) { return -e[get(x)]; }

    bool unite(ll x, ll y)
    { // union by size
        x = get(x), y = get(y);
        if (x == y)
            return false;
        if (e[x] > e[y])
            swap(x, y);
        e[x] += e[y];
        e[y] = x;
        return true;
    }
};
ll find(ll x, vector<ll> &parent)
{
    if (parent[x] != x)
    {
        parent[x] = find(parent[x], parent);
    }
    return parent[x];
}

void union_set(ll x, ll y, vector<ll> &parent, vector<ll> &rank)
{
    ll px = find(x, parent);
    ll py = find(y, parent);

    if (px == py)
        return;

    if (rank[px] < rank[py])
    {
        parent[px] = py;
    }
    else if (rank[px] > rank[py])
    {
        parent[py] = px;
    }
    else
    {
        parent[py] = px;
        rank[px]++;
    }
}
ll countComponents(vector<pair<ll, ll>> &edges)
{
    ll maxNode = 0;
    for (const auto &edge : edges)
    {
        maxNode = max(maxNode, max(edge.first, edge.second));
    }
    vector<ll> parent(maxNode + 1);
    vector<ll> rank(maxNode + 1, 0);
    for (ll i = 0; i <= maxNode; i++)
    {
        parent[i] = i;
    }
    for (const auto &edge : edges)
    {
        union_set(edge.first, edge.second, parent, rank);
    }
    unordered_set<ll> uniqueComponents;
    for (ll i = 0; i <= maxNode; i++)
    {
        bool isPartOfGraph = false;
        for (const auto &edge : edges)
        {
            if (edge.first == i || edge.second == i)
            {
                isPartOfGraph = true;
                break;
            }
        }
        if (isPartOfGraph)
        {
            uniqueComponents.insert(find(i, parent));
        }
    }

    return uniqueComponents.size();
}
void generateCombinations(long long start, long long n, long long k, std::vector<long long> &current, std::vector<std::vector<long long>> &result)
{
    if (current.size() == k)
    {
        result.push_back(current);
        return;
    }

    for (long long i = start; i <= n; ++i)
    {
        current.push_back(i);
        generateCombinations(i + 1, n, k, current, result);
        current.pop_back();
    }
}

// find all combinations of n choose k
std::vector<std::vector<long long>> combine(long long n, long long k)
{
    std::vector<std::vector<long long>> result;
    std::vector<long long> current;
    generateCombinations(1, n, k, current, result);
    return result;
}
void generatePartitions(ll index, ll n, std::vector<std::vector<ll>> &current, std::vector<std::vector<std::vector<ll>>> &result)
{
    if (index > n)
    {
        result.push_back(current);
        return;
    }
    for (auto &group : current)
    {
        group.push_back(index);
        generatePartitions(index + 1, n, current, result);
        group.pop_back();
    }
    current.push_back({index});
    generatePartitions(index + 1, n, current, result);
    current.pop_back();
}
// find all partitions of a set of n elements
std::vector<std::vector<std::vector<ll>>> findAllPartitions(ll n)
{
    std::vector<std::vector<std::vector<ll>>> result;
    std::vector<std::vector<ll>> current;
    generatePartitions(1, n, current, result);
    return result;
}
// idea to be noted: sometimes can generate resources before test cases start and use them in test cases wo that runtime does not added extra
void bfspath(map<ll, vll> &adj, ll start, ll end, vll &path)
{
    queue<ll> que;
    map<ll, bool> visited;
    map<ll, ll> parent;
    que.push(start);
    visited[start] = true;
    while (!que.empty())
    {
        ll node = que.front();
        que.pop();
        for (ll next : adj[node])
        {
            if (!visited[next])
            {
                visited[next] = true;
                parent[next] = node;
                que.push(next);
            }
        }
    }
    if (!visited[end])
    {
        cout << -1 << endl;
        return;
    }
    for (ll at = end; at != -1; at = parent[at])
    {
        path.push_back(at);
    }
    reverse(path.begin(), path.end());
}
vll allsubsetSumsorted(vll &a)
{
    // Use an unordered_set to accumulate distinct sums.
    unordered_set<long long> sums;
    sums.insert(0); // empty subset

    for (ll x : a)
    {
        // collect new sums in a temporary vector to avoid modifying the set
        // while iterating it.
        vll newSums;
        newSums.reserve(sums.size());
        for (auto s : sums)
            newSums.push_back(s + x);

        for (auto ns : newSums)
            sums.insert(ns);
    }

    // move to vector and sort
    vll result(sums.begin(), sums.end());
    sort(result.begin(), result.end());
    return result;
}
ll mod = 1000000007;
ll npr(ll n, ll r)
{
    if (r > n)
        return 0;
    ll numerator = 1;
    for (ll i = 0; i < r; i++)
    {
        numerator = (numerator * (n - i)) % mod;
    }
    return numerator;
}
// Extended GCD: returns gcd(a,b) and finds x,y so that a*x + b*y = gcd(a,b)
ll extgcd(ll a, ll b, ll &x, ll &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    ll x1, y1;
    ll g = extgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// Modular inverse using extended GCD (works for any modulus m)
// Returns inverse in [0, m-1] if exists, otherwise returns -1 to signal no inverse.
ll modInverse_extgcd(ll a, ll m)
{
    a %= m;
    if (a < 0)
        a += m;
    ll x, y;
    ll g = extgcd(a, m, x, y);
    if (g != 1)
        return -1; // inverse doesn't exist
    ll inv = x % m;
    if (inv < 0)
        inv += m;
    return inv;
}
ll chinese_remainder(const vector<ll> &rema, const vector<ll> &mods)
{
    ll M = 1;
    int k = mods.size();
    for (ll m : mods)
        M *= m; // product of all moduli

    ll result = 0;
    for (int i = 0; i < k; i++)
    {
        ll Mi = M / mods[i];                     // M_i = M / m_i
        ll inv = modInverse_extgcd(Mi, mods[i]); // inverse of Mi mod m_i
        if (inv == -1)
            return -1; // if inverse doesn't exist
        result = (result + (__int128)rema[i] * Mi % M * inv % M) % M;
    }
    return (result % M + M) % M; // ensure non-negative
}
ll norm(ll a, ll m)
{
    a %= m;
    if (a < 0)
        a += m;
    return a;
}

// Solve q * x ≡ p (mod m).
// Returns vector of solutions in [0, m-1] (empty if none).
// return p/q mod m
vector<ll> div_mod_solutions(ll p, ll q, ll m)
{
    if (m <= 0)
        return {}; // invalid modulus
    // reduce p and q modulo m for convenience
    p = norm(p, m);
    q = norm(q, m);

    ll x, y;
    ll g = std::gcd(q, m);
    if (p % g != 0)
        return {}; // no solution

    // reduce equation by g
    ll qr = q / g;
    ll pr = p / g;
    ll mr = m / g;

    // find inverse of qr modulo mr using extended gcd
    ll inv, tmp;
    ll gg = extgcd(qr, mr, inv, tmp); // gg should be 1
    // make sure we have inverse (it should because gcd(qr,mr)==1)
    if (gg != 1)
        return {}; // safety
    inv = norm(inv, mr);

    ll x0 = ((__int128)inv * pr) % mr;
    if (x0 < 0)
        x0 += mr;

    // produce the g distinct solutions modulo m:
    vector<ll> sols;
    sols.reserve(g);
    for (ll k = 0; k < g; ++k)
    {
        ll sol = x0 + k * mr; // in range [0, m-1] because x0 in [0,mr-1] and k*mr < m
        sol = norm(sol, m);
        sols.push_back(sol);
    }
    sort(sols.begin(), sols.end());
    return sols;
}
// cnt the number of numbers in [x,y] that are divisible by any of the divisors in v
ll count_in_range(ll x, ll y, vector<ll> v)
{
    if (x > y)
        return 0;
    // Keep only positive divisors <= y
    vector<ll> divs;
    divs.reserve(v.size());
    for (ll d : v)
        if (d > 0 && d <= y)
            divs.push_back(d);
    if (divs.empty())
        return 0;

    sort(divs.begin(), divs.end());
    divs.erase(unique(divs.begin(), divs.end()), divs.end());

    // Remove any divisor that is a multiple of a smaller divisor (redundant)
    vector<ll> filtered;
    for (size_t i = 0; i < divs.size(); ++i)
    {
        bool redundant = false;
        for (size_t j = 0; j < i; ++j)
        {
            if (divs[i] % divs[j] == 0)
            {
                redundant = true;
                break;
            }
        }
        if (!redundant)
            filtered.push_back(divs[i]);
    }
    divs.swap(filtered);
    if (divs.empty())
        return 0;
    if (divs[0] == 1)
        return (y - x + 1); // everything divisible by 1

    auto count_upto = [&](ll up)
    {
        if (up < 1)
            return 0LL;
        ll ans = 0;
        int n = (int)divs.size();

        // DFS with pruning - recursion depth <= n (n typically small after filtering)
        function<void(int, ll, int)> dfs = [&](int idx, ll cur_lcm, int sign)
        {
            for (int i = idx; i < n; ++i)
            {
                // compute lcm(cur_lcm, divs[i]) safely and stop if > up
                ll g = std::gcd(cur_lcm, divs[i]);
                __int128 t = (__int128)(cur_lcm / g) * (__int128)divs[i];
                if (t > up)
                    continue; // pruning: no multiples of this lcm in [1..up]
                ll nlcm = (ll)t;
                ll add = up / nlcm;
                ans += sign * add;
                dfs(i + 1, nlcm, -sign);
            }
        };

        dfs(0, 1LL, +1);
        return ans;
    };

    return count_upto(y) - count_upto(x - 1);
}
// descrete log
ll modpow(ll a, ll e, ll mod)
{
    ll res = 1 % mod;
    a %= mod;
    while (e > 0)
    {
        if (e & 1)
            res = (__int128)res * a % mod;
        a = (__int128)a * a % mod;
        e >>= 1;
    }
    return res;
}

// Extended Euclidean Algorithm
ll egcd(ll a, ll b, ll &x, ll &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    ll x1, y1;
    ll g = egcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return g;
}

// Modular inverse (works even if mod not prime, but gcd(a,mod)=1 required)
ll modinv(ll a, ll mod)
{
    ll x, y;
    ll g = egcd(a, mod, x, y);
    if (g != 1)
        return -1; // inverse doesn't exist
    x %= mod;
    if (x < 0)
        x += mod;
    return x;
}

// Baby-step Giant-step algorithm
ll discrete_log(ll a, ll b, ll m)
{
    a %= m, b %= m;
    if (b == 1 % m)
        return 0;

    ll n = (ll)sqrt(m) + 1;

    // Baby steps: store a^j
    unordered_map<ll, ll> baby;
    ll cur = 1;
    for (ll j = 0; j < n; j++)
    {
        if (!baby.count(cur))
            baby[cur] = j;
        cur = (__int128)cur * a % m;
    }

    // Compute a^(-n)
    ll an = modpow(a, n, m);
    ll inv_an = modinv(an, m);
    if (inv_an == -1)
        return -1; // no solution if inverse doesn't exist

    // Giant steps
    cur = b;
    for (ll i = 0; i <= n; i++)
    {
        if (baby.count(cur))
        {
            ll ans = i * n + baby[cur];
            return ans;
        }
        cur = (__int128)cur * inv_an % m;
    }
    return -1; // no solution
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll t;
    cin >> t;
    while (t--)
    {
    }
    return 0;
}
