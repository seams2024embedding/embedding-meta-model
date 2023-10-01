function r = TDS(src,tar)
    train = {};
    k = 1;
    while length(train) <= 3
        r = nearest(train,src,tar);
        train{k,1} = src{r};
        src(r) = [];
        k = k + 1;
    end
    r = train;
end

function r = nearest(train,src,tar)
    tar = tar(:,1:end-1);
    v_tar = [mean(tar),std(tar)];
    s = [];
    for i=1:length(train)
        s = [s;train{i}(:,1:end-1)];
    end
    for i=1:length(src)
        t = [s;src{i}(:,1:end-1)];
        v_src(i,:) = [mean(t),std(t)];
    end
    for i=1:length(src)
        d(i,1) = norm(v_tar-v_src(i,:));
    end
    [~,r] = min(d);
end