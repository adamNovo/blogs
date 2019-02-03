# 1. Algorithms

Name | Worst | Average | Hidden constants | In place |
 --- | :---: | :---: | :---: | :---: | :---: |
Insertion sort | $\Theta(n^2)$ | $\Theta(n^2)$ | small | yes |
Merge sort | $\Theta(n*log n)$ | $\Theta(n*log n)$ | large | no |
Heap sort | $O(n*log n)$ | - | small | yes |
Quicksort | $\Theta(n^2)$ | $\Theta(n*log n)$ expected | small | yes |
Counting sort | $\Theta(k+n)$ | $\Theta(k+n)$ | large | no |
Radix sort | $\Theta(d*(k+n))$ | $\Theta(d*(k+n))$ | large | no |
Bucket sort | $\Theta(n^2)$ | $\Theta(n)$ | large | no |
Key: k - constant, d - constant


# 2. Sum
Name | Formula |
 --- | :---: |
 Arithmetic | $ \sum_{k=1}^{n} k = \frac{n(n+1)}{2} $ |
 Arithmetic | $ \sum_{k=0}^{n} k^2 = \frac{n(n+1)(2n+1)}{6} $ |
 Arithmetic | $ \sum_{k=0}^{n} k^3 = \frac{n^2(n+1)^2}{4} $ |
 Geometric | $ \sum_{k=0}^{n} x^k = \frac{x^{n+1}-1}{x-1} $ |
 Geometric | $ \sum_{k=0}^{\infty} x^k = \frac{1}{1-x} $, where x < 1 |
 Harmonic | $ \sum_{k=1}^{n} 1/k = ln(n) $ |
 Integrating | $ \sum_{k=0}^{\infty} kx^k = \frac{x}{(1-x)^2} $, where x < 1 |
 
 # 3. Logs
 Exp |  | Equiv |
 --- | :---: | --- |
 $ \log(\prod_{k=1}^{n} a_k) $ | = | $ \sum_{k=1}^{n} log(a_k) $ |
 $ \log_b a $ | = | $ \frac{\log_c a}{\log_c b} $ |

 # 4. Finance
 - [Black Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
$$ C(S_t, t) = N(d_1)*S_t - N(d_2)*Ke^{-r(T-t)} $$
$$ d_1 = \frac{1}{\sigma(T-t)^{1/2}} 
    [\ln(\frac{S_t}{K}) + (r + \frac{\sigma^2}{2}*(T-t)) ]  $$
$$ d_2 = d_1 - \sigma(T-t)^{1/2}  $$
$$ N(\cdot ) = $$
the cumulative distribution function of the standard normal distribution
$$ {\displaystyle S_{t}} $$
the spot price of the underlying asset
$$ {\displaystyle K} $$
the strike price
$$ {\displaystyle r} $$
the risk free rate (annual rate, expressed in terms of continuous compounding)
$$ {\displaystyle \sigma } $$ 
the volatility of returns of the underlying asset
- TBU
    