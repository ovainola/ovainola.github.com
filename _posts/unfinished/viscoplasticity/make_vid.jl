# ================ Plotting 2D =================== #
using PyPlot
fig = figure()
ax = fig["add_subplot"](111, projection="3d")
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

#plot(x_vals, y_vals)
#PyPlot.title("von Mises yield surface in 2D")
#PyPlot.xlabel("Eig Stress 1")
#PyPlot.ylabel("Eig Stress 2")
#grid()
#show()

# ================ Plotting 3D =================== #
n(θ, ϕ) = [sin(θ)*cos(ϕ)
           sin(θ)*sin(ϕ)
           cos(θ)]

m(θ, ϕ, χ) = [-sin(ϕ)*cos(χ)-cos(θ)*cos(ϕ)*sin(χ)
               cos(ϕ)*cos(χ)-cos(θ)*sin(ϕ)*sin(χ)
               sin(θ)*sin(χ)]

w = [sqrt(2/3) * 200 * m(54.735 * pi / 180, 45 * pi/180, x) for x=0:0.15:(2*pi+0.1)]
base_vec =  [1 1 1] / sqrt(3)

for i=-5:5
    tt = [w[x] + vec(base_vec) + 50 * i for x=1:length(w)]
    x = map(x->tt[x][1], collect(1:length(w)))
    y = map(x->tt[x][2], collect(1:length(w)))
    z = map(x->tt[x][3], collect(1:length(w)))
    plot3D(x, y, z, color="blue")
end

tt = [w[x] + vec(base_vec) + 50 * -5 for x=1:length(w)]
x_start = map(x->tt[x][1], collect(1:length(w)))[1:5:end]
y_start = map(x->tt[x][2], collect(1:length(w)))[1:5:end]
z_start = map(x->tt[x][3], collect(1:length(w)))[1:5:end]


tt = [w[x] + vec(base_vec) + 50 * 5 for x=1:length(w)]
x_end = map(x->tt[x][1], collect(1:length(w)))[1:5:end]
y_end = map(x->tt[x][2], collect(1:length(w)))[1:5:end]
z_end = map(x->tt[x][3], collect(1:length(w)))[1:5:end]

for i=1:length(x_start)
    x = [x_start[i], x_end[i]]
    y = [y_start[i], y_end[i]]
    z = [z_start[i], z_end[i]]
    plot3D(x, y, z, color="blue")
end


info("Calculation finished")

# plot the surface
xx = zeros(10, 10)
yy = zeros(10, 10)

for i=1:10
    for j=1:10
        xx[i, j] = (i - 5) * 100
        yy[i, j] = (j - 5) * 100
    end
end

# calculate corresponding z
z = zeros(10, 10)
for i=1:10
    for j=1:10
        z[i, j] = 1 
    end
end

# plot the surface
plot_surface(xx, yy, z, color="blue")
plot(x_vals, y_vals, zeros(length(y_vals)), color="yellow")
axis("equal")
PyPlot.title("von Mises yield surface in 3D")
PyPlot.xlabel("Eig Stress 1")
PyPlot.ylabel("Eig Stress 2")
PyPlot.zlabel("Eig Stress 3")
for i=1:360
    ax["view_init"](30, i)
    PyPlot.draw()
    savefig("$(lpad(i, 3, 0))_plot.png")
end
