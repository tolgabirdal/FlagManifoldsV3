function [dist] = chordal_distance(X, Y, Bs)

k = length(Bs);

dist = 0;
for i=1:k
    Xi = X(1:end,Bs{i});
    Yi = Y(1:end,Bs{i});
    dist = dist + sqrt(length(Bs{i}) - trace(Xi' * Yi * Yi' * Xi));
end

