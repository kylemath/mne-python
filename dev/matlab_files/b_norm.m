% b_norm 
% normalize continuous data from BOXY (unparsed) - function version
%
% © 2015 University of Illinois Board of Trustees, All Rights Reserved.
%
%
% elm 03/10/04 cleaned up print messages, supressed polyfit warnings
%               & chgd "filt' to 'norm'
% elm 2/24/11 added phase outlier cleaning
% elm 7/7/11 saving boxy_hdr, good_norm_mrac_chs & good_mrdc_chs in output file

%cycle	exmux	A-AC	A-DC	A-Ph	B-AC	B-DC	B-Ph	time	group	mark	flag

function b_norm(log_file)
global hdr file_name i_block dc ac ph adc aac aph msg mod_freq id_tags qual
global boxy_hdr

disp('b_norm 6.3');

n_bad_points=12;    % number of anomalous initial data points to zap

in_path=[hdr.data_path hdr.exp_name 'opt\'];

[status,msg] = mkdir([hdr.data_path hdr.exp_name], ['norm00-00']);

out_path=[hdr.data_path hdr.exp_name 'norm00-00\'];

in_file_flg=fopen([in_path file_name]);
out_file_flg=fopen([out_path file_name]);
fclose('all');

% Open the log file if it exists
%
if ~strcmp(log_file,'')
    logid = fopen(log_file,'a');
else logid = 1; end

if ((in_file_flg>0) & (out_file_flg<0))
    
    % read a boxy file
    %
    [boxy_hdr,dc,ac,ph,aux]=read_boxy_hdr([in_path file_name]);
    
    % Fix bad first points (Dennis' problem :)
    %
    for index = 1:n_bad_points
        dc(index,:)=dc(n_bad_points+1,:);
        ac(index,:)=ac(n_bad_points+1,:);
        ph(index,:)=ph(n_bad_points+1,:);
    end
    disp('opened file');
	% fix phase wrap
    % 1st estimate mean phase of point 1 to 50
	% if a pont id more than 90 degrees different from the mean,
    %  add or subtract 360 degrees
    %
	fprintf(logid,'Fixing phase wrap\n');
    
	for i_chan=1:boxy_hdr.n_chans
        if mean(ph(1:50,i_chan)) < 180
            wrapped_pts=find(ph(:,i_chan)>270);
            ph(wrapped_pts,i_chan)=ph(wrapped_pts,i_chan)-360;
        else
            wrapped_pts=find(ph(:,i_chan)<90);
            ph(wrapped_pts,i_chan)=ph(wrapped_pts,i_chan)+360;
        end
	end
    
    % Detrend (polynomial) phase data (fix Emily's problem ;-)
    % (also sets mean = 0 !)
    %
    fprintf(logid,'Detrending\n');
    y=1:boxy_hdr.n_points;
    x=y';
    warning off MATLAB:polyfit:RepeatedPointsOrRescale
    
    for i_ch=boxy_hdr.n_chans
        poly_coeffs = polyfit(x,ph(:,i_ch),3);   % 3 => 3rd order
        tmp_ph = ph(:,i_ch)-polyval(poly_coeffs,x);
        ph(:,i_ch)=tmp_ph;
    end
    
    % remove mean here so STD threshold makes sense
    %
    mrph=mean(ph);
    for i_chan=1:boxy_hdr.n_chans
        ph(:,i_chan)=(ph(:,i_chan)-mrph(i_chan));
    end
    
    % remove phase (delay) outliers
    %
    hdr.qual.ph_out_thr=3; % always set to "3" per Kathy & Gabriele Oct 12 2012
    SDph=std(ph(n_bad_points:end,:));
    ph_out_thr=hdr.qual.ph_out_thr; 
    for i_chan=1:boxy_hdr.n_chans
        outliers=find(abs(ph(:,i_chan))>ph_out_thr*SDph(i_chan));
        if length(outliers)>0
            if outliers(1)==1;outliers=outliers(2:end);end % can't interp 1st pt
            if outliers(end)==boxy_hdr.n_points % can't interp last pt
                outliers=outliers(1:end-1);
            end
            n_ph_out(i_chan)=length(outliers);
            for i_pt=1:n_ph_out(i_chan)
                j_pt=outliers(i_pt);
                ph(j_pt,i_chan)=(ph(j_pt-1,i_chan)+ph(j_pt+1,i_chan))/2;
            end
        end
    end
    
	% Compute means
	%
	mrdc=mean(dc);
%     dcneg=find(mrdc<0); % find chs w -dc
%     mrdc(dcneg)=0; % and set them to 0
	mrac=mean(ac);
	mrph=mean(ph);
    
    % stash raw means in hdr
    %
    hdr.mrdc=mrdc;
    hdr.mrac=mrac;
    hdr.mrph=mrph;
    
    % apply DC & AC quality thresholds if desired
    % zeroing 'bad' chans
    %
    good_norm_mrac_chs=[];
    good_norm_mrdc_chs=[];
    if qual.InitQualChks_flg==1
        good_norm_mrac_chs=find(mrac>=hdr.qual.norm_mrac_thr);
        bad_mrac_chs=find(mrac<hdr.qual.norm_mrac_thr);
        good_norm_mrdc_chs=find(mrdc>=hdr.qual.norm_mrdc_thr);
        bad_mrdc_chs=find(mrdc<hdr.qual.norm_mrdc_thr);

        % set bad mrdc & mrac chans = 0
        %
        dc(:,bad_mrdc_chs)=0;
        ac(:,bad_mrdc_chs)=0;
        ph(:,bad_mrdc_chs)=0;
        dc(:,bad_mrac_chs)=0;
        ac(:,bad_mrac_chs)=0;
        ph(:,bad_mrac_chs)=0;
    else % use all chans
        good_norm_mrac_chs=1:boxy_hdr.n_chans;
        good_norm_mrdc_chs=1:boxy_hdr.n_chans;
    end % if InitQualChks_flg
    
    % Normalize, and remove mean ('1') for filtering
	%
	fprintf(logid,'Rescaling (''Normalizing'')\n');
	for i_chan=1:boxy_hdr.n_chans
%         if mrdc(i_chan) ~= 0
			dc(:,i_chan)=dc(:,i_chan)/mrdc(i_chan)-1;
%         end
%         if mrac(i_chan) ~= 0
            ac(:,i_chan)=ac(:,i_chan)/mrac(i_chan)-1;
%         end
        ph(:,i_chan)=(ph(:,i_chan)-mrph(i_chan));
    end
    
	ph=1e12*ph/(360*mod_freq); % convert phase to ps
    
    % save the cleaned up and normalized data
    % could save outliers, would need to make cell array of channels...
    %
    hdr_norm=hdr; % save hdr WITH UNIQUE NAME!
	msg=sprintf('Saving %s...\n',[out_path file_name]);
    fprintf(logid,'Saving %s\n',[out_path file_name]);
   	save([out_path file_name],'dc','ac','ph','mrdc','mrac','mrph','SDph',...
        'boxy_hdr','aux',...
        'poly_coeffs','id_tags','mod_freq','hdr_norm',...
        'good_norm_mrac_chs','good_norm_mrdc_chs','-mat');
% 'mark','boxy_flags','aux1','aux2','digaux','aux_data',

else % if various files exist or don't...
    if (in_file_flg<0)
        msg=sprintf('%s does not exist.\n',[in_path file_name]);
        fprintf(logid,'%s does not exist.\n',[in_path file_name]);
    end
    if (out_file_flg>0)
        msg=sprintf('%s already exists!\n',[out_path file_name]);
        fprintf(logid,'%s already exists!\n',[out_path file_name]);
    end
end

if logid ~= 1; fclose(logid); end
