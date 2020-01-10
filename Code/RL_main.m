
DF = 0.9;
% DF: Discount Factor
% EP: Exploration Probability
% LR: Learning Rate

trial_times = 3000;
step_max = 300;

Q = zeros(100,4);
% Initialize Q-function
for j = 1:trial_times
    s1 = 1; % Initial state: 1
    % s1: the current state
    % s2: the next state
    for k = 1:step_max
        EP = 100 / (100 + k);
        EP(EP>1) = 1;
        LR = EP;
        [~, ept] = max(qevalreward(s1,:));
        % ept: exploitation
        % epr: exploration
        % a: action
        if rand() < EP
            epr = find(qevalreward(s1,:)~=-1);
            epr(epr==ept) = [];
            a = epr(randi(length(epr)));
        else
            a = ept;
        end
        if a == 1
            s2 = s1 - 1;
        elseif a == 2
            s2 = s1 + 10;
        elseif a == 3
            s2 = s1 + 1;
        elseif a == 4
            s2 = s1 - 10;
        end
        Q(s1,a) = Q(s1,a)+LR*(qevalreward(s1,a)+DF*max(Q(s2,:))-Q(s1,a));
        % Update Q-function
        s1 = s2;
        if s2==100 || EP<0.005
            break
        end
    end
end

[~, policy] = max(Q,[],2);
s = 1;
k = 0;
Rt = 0;
% Rt: total reward
while s~=100 && k<step_max
    a = policy(s);
    Rt = Rt + DF^k*qevalreward(s,a);
    if a == 1
        s = s - 1;
    elseif a == 2
        s = s + 10;
    elseif a == 3
        s = s + 1;
    elseif a == 4
        s = s - 10;
    end
    k = k + 1;
end
if s ~= 100
    Rt = -1;
end

if Rt > 0
    fprintf("The optimal policy is shown as the Trajectory Plotting.\n");
    fprintf("The total reward of the optimal policy: %.4f\n",Rt);
    
% Trajectory Plotting
    figure('color','w');
    hold on;
    [X,Y] = meshgrid(0:10,0:10);
    plot(X,Y,'b',X',Y','b');
    X = 0.5;
    Y = 9.5;
    s = 1;
    s_record = s;
    while s~=100
        if policy(s) == 1
            plot(X,Y,'r^','markersize',10);
            Y = Y + 1;
            s = s - 1;
        elseif policy(s) == 2
            plot(X,Y,'r>','markersize',10);
            X = X + 1;
            s = s + 10;
        elseif policy(s) == 3
            plot(X,Y,'rv','markersize',10);
            Y = Y - 1;
            s = s + 1;
        elseif policy(s) == 4
            plot(X,Y,'r<','markersize',10);
            X = X - 1;
            s = s - 10;
        end
        s_record = [s_record;s];
    end
    qevalstates = s_record;
    plot(X,Y,'r*','markersize',10);
    title({'Trajectory Plotting';['Total Reward: ',num2str(Rt)]});
    axis equal;
    axis off;
    hold off;
end
