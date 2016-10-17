using PyPlot
stress_y =  200.0

function vm_upper(a, c)
    vals = f(a[1], a[2], c)
    vm(vals[1], vals[2], 200)
end
vm(a,b) = sqrt(a^2 - a*b + b^2) - stress_y
f(m,c) = [600*cos(c) 600*sin(c)].*m
x_vals = []
max_iter = 100
y_vals = []
for i=0:0.1:(2*pi+0.3)
    wf(x) = f(x, i)
    t = 0.01
    step = 2
    merkki = -1
    s11, s22 = wf(t)
    ii = 0
    while (abs(vm(s11, s22)) > 1e-7) && ii < max_iter
        val = vm(s11, s22)
        if sign(val) != merkki
            merkki *= -1
            step *= -0.5
        end
        t += step
        s11, s22 = wf(t)
        ii += 1
    end
    push!(x_vals, s11)
    push!(y_vals, s22)
end

plot(x_vals, y_vals)
PyPlot.title("von Mises yield surface in 2D")
PyPlot.xlabel("Eig Stress 1")
PyPlot.ylabel("Eig Stress 2")
grid()
show()
