# Include necessary modules for I/O
using DelimitedFiles

# Define the `myrandom` function
function myrandom(r,a,b)
    r = mod(r*41475557, 1)
    return (r, a + r * (b - a))
end

# Initialize variables
size   = 2000
dens   = 0.75
seeds  = [2,3,4,5,6]
n      = size - 1
dvert  = 2

# Loop to generate the files
for igen = 1:5
    myseed = seeds[igen]
    r = (4 * myseed + 1) / (16384 * 16384)

    Fc = zeros(n+1, n+1)
    Fl = zeros(n+1, n+1)
    F  = zeros(n+1, n+1)

    # Fill the matrices Fc, Fl, and F according to the provided logic
    for i = 1:n
        for j = (i+1):(n+1)
            (r, num) = myrandom(r, 0, 1)
            if num < dens
                (r, num) = myrandom(r, 0.0, 10.0)
                Fc[i, j] = Fc[j, i] = num
            else
                (r, num) = myrandom(r, -10.0, 0.0)
                Fc[i, j] = Fc[j, i] = num
            end
        end
    end

    for i = 1:(n+1)
        (r, num) = myrandom(r, 0, dvert)
        Fl[i, i] = num
    end

    for i = 1:n
        for j = (i+1):(n+1)
            Fl[i, j] = Fl[j, i] = 0.5 * (Fl[i, i] + Fl[j, j])
        end
    end

    for i = 1:(n+1)
        for j = i:(n+1)
            F[i, j] = F[j, i] = Fl[i, j] - Fc[i, j]
        end
    end

    # Generate the filename and write the matrix to a file
    filename = "Problem_$(n+1)x$(n+1)_$(dens)_$(igen).txt"
    writedlm(filename, F)
end

# End the script without returning anything
nothing
