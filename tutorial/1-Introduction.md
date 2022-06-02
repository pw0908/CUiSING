# Introduction to Julia
Julia is often described as a language with _'Python-like syntax and C-like speeds'_ and this isn't too far from the truth. At the surface, Julia does present itself more like a compilled language disguised as an interpreted language. For example, let us first write a code that fills an array with the sum of even numbers from 0 to 100 in different languages. Firstly, python:
```python
A = [0]
for i in range(1,101):
    if i%2 == 0:
        A.append(A[-1]+i)
```
In julia:
```julia
A = [0]
for i in range(1,100)
    if i%2 == 0
        append!(A,A[end]+i)
    end
end
```
As we can see, the syntax is quite close to python, with one key difference: julia is index-1 while python (and C++) are index-0. However, julia does not have any strict indentation rules like python does. Nevertheless, in theory, you could copy over code from python into julia and only have to change a few things to get it to work. This is one of the selling points of julia.

## Defining functions
Let us say we want to define a function which performs the following:
$$
f(x,y) = \sqrt{x^2+y^2}
$$
In python, we would code this as:
```python
def f(x, y):
    return (x**2+y**2)**0.5
```
In julia, this would be very similar:
```julia
function f(x, y)
    return sqrt(x^2+y^2)
end
```
We can also simply write:
```julia
f(x, y) = √(x^2+y^2)
```
For the same result. Note that, in the above, we have used the Unicode character for our square root. One thing that the julia community strives to push forward is the use of Unicode characters in code to bring it as close as possible to the true mathematical expressions (Unicode characters for greek characters are also supported). 

Another function type (shown above in the form of `append!`) is one which modifies its inputs. In the case of `append!`, we modified the input by adding an element to it. However, if we want to define a function which transforms its input to the square of itself, we would simply need to do the following:
```julia
function square!(x,y)
    y = x^2
end
```
We can see the effect of this:
```julia
julia> x=2
2

julia> square!(x,y)
4

julia> x
4
```
It is functions like these that allow Julia to behave more like a compilled language than an interpreted language. Interested readers are recommended to look at information regarding Julia's [_just-in-time_ compiller](https://kipp.ly/blog/jits-intro/). 
## Broadcasting
One useful feature of julia is the ability to broadcast variables. Let us take the case we have have two arrays for `x` and `y` that we wish to apply the function `f` to. In most languages, we would either vectorise the code or use a for loop. However, in julia, one can simply use what is referred to as broadcasting:
```julia
julia> x = rand(100)
julia> y = rand(100)

julia> z = f.(x,y)
```
In the above, using `.` between the function name and inputs, we have told julia to broadcast the function `f` across `x` and `y`. This requires that both `x` and `y` have the same length, or that one of them is length 1.