clear;
load task1.mat;

DF = 0.9;
% DF: Discount Factor: 0.5 or 0.9
EP_type = 4;
% EP: Exploration Probability: 4 types
% LR: Learning Rate: the same as EP

run_times = 10;
trial_times = 3000;
step_max = 100;

Q_runs = zeros(100,4,run_times);
time_record = zeros(1,run_times);
policy = zeros(100,run_times);
Rt = zeros(1,run_times);
% Rt: total reward

for i = 1:run_times
    tic;
    Q = zeros(100,4);
    % Initialize Q-function
    for j = 1:trial_times
        s1 = 1; % Initial state: 1
        % s1: the current state
        % s2: the next state
        for k = 1:step_max
            if EP_type == 1
                EP = 1/k;
            elseif EP_type == 2
                EP = 100/(100+k);
            elseif EP_type == 3
                EP = (1+log(k))/k;
            elseif EP_type == 4
                EP = (1+5*log(k))/k;
            end
            EP(EP>1) = 1;
            LR = EP;
            [~, ept] = max(reward(s1,:));
            % ept: exploitation
            % epr: exploration
            % a: action
            if rand() < EP
                epr = find(reward(s1,:)~=-1);
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
            Q(s1,a) = Q(s1,a)+LR*(reward(s1,a)+DF*max(Q(s2,:))-Q(s1,a));
            % Update Q-function
            s1 = s2;
            if s2==100 || EP<0.005
                break
            end
        end
    end
    time_record(i) = toc;
    Q_runs(:,:,i) = Q;
    [~, policy(:,i)] = max(Q_runs(:,:,i),[],2);
    s = 1;
    k = 0;
    while s~=100 && k<step_max
        a = policy(s,i);
        Rt(i) = Rt(i) + DF^k*reward(s,a);
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
        Rt(i) = -1;
    end
end
Reach = find(Rt > 0);
Reach_num = length(Reach);
exec_time = mean(time_record(Reach));
fprintf("No. of goal-reached runs: %d\n",Reach_num);

if Reach_num > 0
    fprintf("Execution time: %.2f sec\n",exec_time);
    fprintf("The optimal policy is shown as the Trajectory Plotting.\n");
    fprintf("The total reward of the optimal policy: %.4f\n",Rt(1));
    
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
        if policy(s,1) == 1
            plot(X,Y,'r^','markersize',10);
            Y = Y + 1;
            s = s - 1;
        elseif policy(s,1) == 2
            plot(X,Y,'r>','markersize',10);
            X = X + 1;
            s = s + 10;
        elseif policy(s,1) == 3
            plot(X,Y,'rv','markersize',10);
            Y = Y - 1;
            s = s + 1;
        elseif policy(s,1) == 4
            plot(X,Y,'r<','markersize',10);
            X = X - 1;
            s = s - 10;
        end
        s_record = [s_record;s];
    end
    plot(X,Y,'r*','markersize',10);
    title({'Trajectory Plotting';['Total Reward: ',num2str(Rt(1))]});
    axis equal;
    axis off;
    hold off;
end
