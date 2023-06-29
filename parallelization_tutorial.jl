using Base.Threads

# Print the number of threads to be used
println("Number of threads: ", Threads.nthreads())

# Simple example of parallelizing a for loop
a = zeros(9)
@threads for i in eachindex(a)
    a[i] = Threads.threadid()
end
println(a)

# Use atomic operations when the threads will access and modify a given variable to avoid race problems
b = Ref(0)
@threads for i in 1:1000
    b[] += 1
end
println(b)

c = Atomic{Int}(0);
@threads for id in 1:1000
    atomic_add!(c, 1)
end
println(c)
