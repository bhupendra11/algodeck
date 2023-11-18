# Math

## a = a property

Reflexive

[#math](math.md)

## If a = b and b = c then a = c property

Transitive 

[#math](math.md)

## If a = b then b = a property

Symmetric

[#math](math.md)

## Logarithm definition

Inverse function to exponentiation

- log2(1) = 0
- log2(2) = 1
- log2(4) = 2
- log2(8) = 3
- log2(16) = 4
- etc.

[#math](math.md)

## Median of a sorted array

If odd: middle value

If even: average of the two middle values (1, 2, 3, 4 => (2 + 3) / 2 = 2.5)

[#math](math.md)

## n-choose-k problems

From a set of n items, choose k items with 0 <= k <= n

P(n, k)

Order matters: n! / (n - k)! // How many permutations

Order does not matter: n! / ((n - k)! k!) // How many combinations

[#math](math.md)

## Probability: P(a ∩ b) // inter

P(a ∩ b) = P(a) * P(b)

[#math](math.md)

## Probability: P(a ∪ b) // union

P(a ∪ b) = P(a) + P(b) - P(a ∩ b)

[#math](math.md)

## Probability: Pb(a) // probability of a knowing b

Pb(a) = P(a ∩ b) / P(b)

[#math](math.md)

## Prime numbers from 1 - n

### Java O(n) | sieve of eratosthenes

Operations: 
     N/2 + N/3 + N/5+ N/7 + N/11 + ...........N/N          =  Nlog(logN)

So TC:  O( Nlog(logn) )
```java
    public int countPrimes(int n) {
        
        boolean primes[] = new boolean[n+1];
        //consider all as primes initially
        Arrays.fill(primes,true);  

        //only need to check till sqrt as every factor is in pairs
        for(int i=2;i<Math.sqrt(n);i++){ 
            if(primes[i]){
                //mark all multiples as non prime
                for(int j =2*i ;j<n; j+=i){
                    primes[j] = false;
                }
            }
        }
        int count =0;
        for(int i=2;i<n;i++){
            if(primes[i])count++;
        }
        return count;
    }
```
[#math](math.md)

## IsPrime

Did you forget to think of <= instead of <

```java
   private static boolean isPrime(int n) {
        if(n<2)return false;
        for(int i=2 ;i<=Math.sqrt(n);i++){
            if(n%i ==0)return false;
        }
        return true;
    }
```

[#math](math.md)

## Euclidien Algorithm

It is used to find GCD (Greatest Common Divisor) aka HCF(Highest Common Factor)
for two numbers.
    The algorithm basically tries to reduce both numbers by using %(modulo) will one of
    them becomes zero.
The other remaining number is the GCD/HCF

```java

int gcd(int a, int b){
    if(a == 0)
         return b;
    return gcd(b%a, a);
}
```
TC :``` O(log(min(a,b)))```

[#math](math.md)

## Binary Expantiation

Binary Exponentiation is a technique of computing a number raised to some quantity, which can be as small as 
0 or as large as ```10^18```  in a fast and efficient manner. It utilises the property of exponentiation and the fact that any number can be represented as sum of powers of two depending on its binary notation to get an edge over the naive brute force method.

Time complexity -  ```O(log(b))```

Algorithm: At each step do 2 things : 1. Square the base , 2. half the exponent

#### Code
```java
long power(long a,long b){
    long result = 1;
    while(b >0){ 
        if(b%2 ==1){   //for odd b case, one expo would be lost by division further so use here
            result *= a;
        }
        a *= a ;  //square the base
        b /= 2; // half the exponent
    }
    return result;
}
```

 ## Catalan Numbers

```math
C_n = \frac{1}{n+1} \binom{2n}{n}
```