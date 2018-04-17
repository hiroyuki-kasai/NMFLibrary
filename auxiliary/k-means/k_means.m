function cluster_labels = k_means(data, centers, num_clusters)
%K_MEANS Euclidean k-means clustering algorithm.
%
%   Input    : data           : N-by-D data matrix, where N is the number of data,
%                               D is the number of dimensions
%              centers        : K-by-D matrix, where K is num_clusters, or
%                               'random', random initialization, or
%                               [], empty matrix, orthogonal initialization
%              num_clusters   : Number of clusters
%
%   Output   : cluster_labels : N-by-1 vector of cluster assignment
%
%   Reference: Dimitrios Zeimpekis, Efstratios Gallopoulos, 2006.
%              http://scgroup.hpclab.ceid.upatras.gr/scgroup/Projects/TMG/

%
% Parameter setting
%
iter = 0;
qold = inf;
threshold = 0.001;

%
% Check if with initial centers
%
if strcmp(centers, 'random')
  disp('Random initialization...');
  centers = random_init(data, num_clusters);
elseif isempty(centers)
  disp('Orthogonal initialization...');
  centers = orth_init(data, num_clusters);
end

%
% Double type is required for sparse matrix multiply
%
data = double(data);
centers = double(centers);

%
% Calculate the distance (square) between data and centers
%
n = size(data, 1);
x = sum(data.*data, 2)';
X = x(ones(num_clusters, 1), :);
y = sum(centers.*centers, 2);
Y = y(:, ones(n, 1));
P = X + Y - 2*centers*data';

%
% Main program
%
while 1
  iter = iter + 1;

  % Find the closest cluster for each data point
  [val, ind] = min(P, [], 1);
  % Sum up data points within each cluster
  P = sparse(ind, 1:n, 1, num_clusters, n);
  centers = P*data;
  % Size of each cluster, for cluster whose size is 0 we keep it empty
  cluster_size = P*ones(n, 1);
  % For empty clusters, initialize again
  zero_cluster = find(cluster_size==0);
  if length(zero_cluster) > 0
    disp('Zero centroid. Initialize again...');
    centers(zero_cluster, :)= random_init(data, length(zero_cluster));
    cluster_size(zero_cluster) = 1;
  end
  % Update centers
  centers = spdiags(1./cluster_size, 0, num_clusters, num_clusters)*centers;

  % Update distance (square) to new centers
  y = sum(centers.*centers, 2);
  Y = y(:, ones(n, 1));
  P = X + Y - 2*centers*data';

  % Calculate objective function value
  qnew = sum(sum(sparse(ind, 1:n, 1, size(P, 1), size(P, 2)).*P));
  %mesg = sprintf('Iteration %d:\n\tQold=%g\t\tQnew=%g', iter, full(qold), full(qnew));
  %disp(mesg);

  % Check if objective function value is less than/equal to threshold
  if threshold >= abs((qnew-qold)/qold)
    mesg = sprintf('\nkmeans converged!');
    %disp(mesg);
    break;
  end
  qold = qnew;
end

cluster_labels = ind';


%-----------------------------------------------------------------------------
function init_centers = random_init(data, num_clusters)
%RANDOM_INIT Initialize centroids choosing num_clusters rows of data at random
%
%   Input : data         : N-by-D data matrix, where N is the number of data,
%                          D is the number of dimensions
%           num_clusters : Number of clusters
%
%   Output: init_centers : K-by-D matrix, where K is num_clusters
rand('twister', sum(100*clock));
init_centers = data(ceil(size(data, 1)*rand(1, num_clusters)), :);

function init_centers = orth_init(data, num_clusters)
%ORTH_INIT Initialize orthogonal centers for k-means clustering algorithm.
%
%   Input : data         : N-by-D data matrix, where N is the number of data,
%                          D is the number of dimensions
%           num_clusters : Number of clusters
%
%   Output: init_centers : K-by-D matrix, where K is num_clusters

%
% Find the num_clusters centers which are orthogonal to each other
%
Uniq = unique(data, 'rows'); % Avoid duplicate centers
num = size(Uniq, 1);
first = ceil(rand(1)*num); % Randomly select the first center
init_centers = zeros(num_clusters, size(data, 2)); % Storage for centers
init_centers(1, :) = Uniq(first, :);
Uniq(first, :) = [];
c = zeros(num-1, 1); % Accumalated orthogonal values to existing centers for non-centers
% Find the rest num_clusters-1 centers
for j = 2:num_clusters
  c = c + abs(Uniq*init_centers(j-1, :)');
  [minimum, i] = min(c); % Select the most orthogonal one as next center
  init_centers(j, :) = Uniq(i, :);
  Uniq(i, :) = [];
  c(i) = [];
end
clear c Uniq;
