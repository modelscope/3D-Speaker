#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys

def readlist_fileid2elems(inreflist):
    with open(inreflist,"r") as fresfp:
        reflines = fresfp.readlines()
    fileid = {}
    fileid2elems = {}
    fid = 0
    for refline in reflines:
        with open(refline.strip(),"r") as frefp:
            refcontents = frefp.readlines()
        elems = []
        for refcontent in refcontents:
            refsegs = refcontent.strip().split()
            if refsegs[1] not in fileid:
                fname = refsegs[1]
                fileid[refsegs[1]] = fid
                fid = fid + 1
            elem = []
            elem.append(refsegs[1])
            elem.append(float(refsegs[3]))
            elem.append(float(refsegs[4]))
            if refsegs[7].startswith("spk") == True:
                spkid = int(refsegs[7].replace("spk","")) +1
            else:
                spkid = int(refsegs[7])
            elem.append(spkid)
            elems.append(elem)
        if fname in fileid2elems:
            print("error, %s in %s has duplicate file id: %s vs %s" % (fname,refcontent,fileid2elems[fname],elems))
        if fid > 0:
            fileid2elems[fname] = elems
    return fileid2elems

def readfile_fileid2elems(sysfile):
    fileid = {}
    fileid2elems = {}
    fid = 0
    with open(sysfile,"r") as fsysp:
        syscontents = fsysp.readlines()
    elems = []
    for syscontent in syscontents:
        syssegs = syscontent.strip().split()
        if syssegs[1] not in fileid:
            if fid > 0:
                fileid2elems[fname] = elems
            fname = syssegs[1]
            elems = []
            fileid[syssegs[1]] = fid
            fid = fid + 1
        elem = []
        elem.append(syssegs[1])
        elem.append(float(syssegs[3]))
        elem.append(float(syssegs[4]))
        elem.append(int(syssegs[7]))
        elems.append(elem)
    fileid2elems[fname] = elems
    return fileid2elems

def res_err_time(st,et, sid, ref_list):
    err_time = 0
    for elem_ref in ref_list:
        st_ref = elem_ref[0]
        et_ref = elem_ref[0] + elem_ref[1]
        id_ref = elem_ref[2]
        if sid != id_ref:
            if et < et_ref and et > st_ref:
                err_time += et - max(st,st_ref)
            else:
                if st > st_ref and st < et_ref:
                    err_time += min(et_ref,et) - st
                if st < st_ref and et > et_ref:
                    err_time += (et_ref - st_ref)
    return err_time


def eval_elems(fileid2elems_sys,fileid2elems_ref):
    fileid2segs_sys = {}
    fileid2time_sys = {}
    file2spknum_sys = {}
    file2spknum_ref = {}
    file2spkid_sys = {}
    file2spkid_ref = {}
    for (fileid_sys, elems_sys) in fileid2elems_sys.items():
        if fileid_sys not in fileid2elems_ref:
            print("error,there is no fileid_sys in ref rttm: %s" % (fileid_sys))
        else:
            spknum_sys = 0
            for elem in elems_sys:
                if elem[0] != fileid_sys:
                    print("error, elem is not in fileid_sys")
                if elem[3] > spknum_sys:
                    spknum_sys = elem[3]
            file2spknum_sys[fileid_sys] = spknum_sys
            spkelems_sys = []
            spktt_sys = []
            for i in range(spknum_sys):
                spkelems_sys_spk = []
                total_sys_time = 0
                for elem in elems_sys:
                    if elem[3] == i+1:
                        spkelem = []
                        spkelem.append(elem[1])
                        spkelem.append(elem[2])
                        spkelem.append(elem[3])
                        spkelems_sys_spk.append(spkelem)
                        total_sys_time += float(elem[2])
                spkelems_sys.append(spkelems_sys_spk)
                spktt_sys.append(total_sys_time)
            b = enumerate(spktt_sys)
            spk_time_sys = []
            spk_elem_sys = []
            spk_id_sys = []
            for e in b:
                spk_time_sys.append(e[1])
                spk_elem_sys.append(spkelems_sys[e[0]])
                spk_id_sys.append(e[0]+1)
            fileid2segs_sys[fileid_sys] = spk_elem_sys
            fileid2time_sys[fileid_sys] = spk_time_sys
            file2spkid_sys[fileid_sys] = spk_id_sys

    fileid2segs_ref = {}
    fileid2time_ref = {}
    for (fileid_ref, elems_ref) in fileid2elems_ref.items():
        if fileid_ref not in fileid2elems_sys:
            print("error,there is no fileid_ref in sys rttm: %s" % (fileid_ref))
        else:
            elems_ref = fileid2elems_ref[fileid_ref]
            spknum_ref = 0
            for elem in elems_ref:
                if elem[0] != fileid_ref:
                    printf("error, elem is not in fileid_sys")
                if elem[3] > spknum_ref:
                    spknum_ref = elem[3]
            file2spknum_ref[fileid_ref] = spknum_ref
            spkelems_ref = []
            spktt_ref = []
            for i in range(spknum_ref):
                spkelems_ref_spk = []
                total_ref_time = 0
                for elem in elems_ref:
                    if elem[3] == i + 1:
                        spkelem = []
                        spkelem.append(elem[1])
                        spkelem.append(elem[2])
                        spkelem.append(elem[3])
                        spkelems_ref_spk.append(spkelem)
                        total_ref_time += elem[2]
                spkelems_ref.append(spkelems_ref_spk)
                spktt_ref.append(total_ref_time)
            b = enumerate(spktt_ref)
            spk_time_ref = []
            spk_elem_ref = []
            spk_id_ref = []
            for e in b:
                spk_time_ref.append(e[1])
                spk_elem_ref.append(spkelems_ref[e[0]])
                spk_id_ref.append(e[0] + 1)
            fileid2segs_ref[fileid_ref] = spk_elem_ref
            fileid2time_ref[fileid_ref] = spk_time_ref
            file2spkid_ref[fileid_ref] = spk_id_ref

    if len(fileid2segs_ref) != len(fileid2segs_sys):
        print("error, length of filedi2 segs_ref != fileid2segs_sys")
    else:
        file2core_time = {}
        file2total_time = {}
        file2sysid = {}
        file2refid = {}
        for (fileid_sys, elems_sys) in fileid2segs_sys.items():
            spknum_ref = file2spknum_ref[fileid_sys]
            spknum_sys = file2spknum_sys[fileid_sys]
            ori_spkid_ref = file2spkid_ref[fileid_sys]
            ori_spkid_sys = file2spkid_sys[fileid_sys]
            elems_ref = fileid2segs_ref[fileid_sys]
            ref_idmap = []
            for i in range(spknum_ref):
                ref_idmap.append(1)
            spkid_cor_time = []
            spkid_total_time = []
            spkid_sys = []
            spkid_ref = []
            cor_time_ids = []
            for id_sys in ori_spkid_sys:
                max_time = -1
                max_id = -1
                cor_time_id = []
                for id_ref in ori_spkid_ref:
                    cor_time = 0
                    cor_sys_seg = []
                    for ii in range(len(elems_sys)):
                        elem_spk_sys = elems_sys[ii]
                        if elem_spk_sys[0][2] == id_sys:
                            for elem_ii in elem_spk_sys:
                                st_sys = elem_ii[0]
                                et_sys = elem_ii[0] + elem_ii[1]
                                err_time = 0
                                for i in range(len(elems_ref)):
                                    elem_spk_ref = elems_ref[i]
                                    for elem_i in elem_spk_ref:
                                        st_ref = elem_i[0]
                                        et_ref = elem_i[0] + elem_i[1]
                                        if elem_spk_ref[0][2] != id_ref or ref_idmap[id_ref-1] != 1:
                                            if et_ref > st_sys and st_ref < st_sys:
                                                err_time += min(et_sys, et_ref) - st_sys
                                            else:
                                                if et_ref >= et_sys and st_ref < et_sys:
                                                    err_time += et_sys - max(st_sys, st_ref)
                                                else:
                                                    if st_ref >= st_sys and et_ref <= et_sys:
                                                        err_time += et_ref - st_ref
                                cor_time += elem_ii[1] - err_time
                    cor_time_id.append(cor_time)
                cor_time_ids.append(cor_time_id)
                spkid_time = 0
                for ii in range(len(elems_sys)):
                    elem_spk_sys = elems_sys[ii]
                    if elem_spk_sys[0][2] == id_sys:
                        for elem_ii in elem_spk_sys:
                            spkid_time += elem_ii[1]
                spkid_total_time.append(spkid_time)

            max_time = -1
            max_id_ori = -1
            max_id_ref = -1
            spkid_total_time_new = []
            ori_spkid_sys_label = []
            for i in range(len(ori_spkid_sys)):
                ori_spkid_sys_label.append(-1)
            for t in range(len(ori_spkid_sys)):
                for i in range(len(ori_spkid_sys)):
                    for j in range(len(ori_spkid_ref)):
                        if cor_time_ids[i][j] > max_time:
                            max_id_ori = i
                            max_id_ref = j
                            max_time = cor_time_ids[i][j]
                if max_time != -1:
                    ori_spkid_sys_label[max_id_ori] = 1
                    spkid_sys.append(max_id_ori)
                    spkid_cor_time.append(max_time)
                    spkid_total_time_new.append(spkid_total_time[max_id_ori])
                spkid_ref.append(max_id_ref)
                for i in range(len(ori_spkid_sys)):
                    cor_time_ids[i][max_id_ref] = -1
                for j in range(len(ori_spkid_ref)):
                    cor_time_ids[max_id_ori][j] = -1
                max_time = -1
                max_id_ori = -1
                max_id_ref = -1
            for i in range(len(ori_spkid_sys_label)):
                if ori_spkid_sys_label[i]==-1:
                    spkid_sys.append(i)
                    spkid_cor_time.append(0)
                    spkid_total_time_new.append(spkid_total_time[i])

            file2core_time[fileid_sys] = spkid_cor_time
            file2total_time[fileid_sys] = spkid_total_time_new
            file2sysid[fileid_sys] = spkid_sys
            file2refid[fileid_sys] = spkid_ref
    return file2core_time,file2total_time,file2sysid,file2refid,file2spknum_sys,file2spknum_ref

def eval_elems_pur(fileid2elems_sys,fileid2elems_ref,colar,file2sysid,file2refid):
    fileid2segs_sys = {}
    fileid2time_sys = {}
    file2spknum_sys = {}
    file2spknum_ref = {}
    file2spkid_sys = {}
    file2spkid_ref = {}
    file2cor_segnum = {}
    for (fileid_sys, elems_sys) in fileid2elems_sys.items():
        if fileid_sys not in fileid2elems_ref:
            print("seg pur error,there is no fileid_sys in ref rttm: %s", fileid_sys)
        else:
            spknum_sys = 0
            for elem in elems_sys:
                if elem[0] != fileid_sys:
                    printf("error, elem is not in fileid_sys")
                if elem[3] > spknum_sys:
                    spknum_sys = elem[3]
            file2spknum_sys[fileid_sys] = spknum_sys
            spkelems_sys = []
            spktt_sys = []
            for i in range(spknum_sys):
                spkelems_sys_spk = []
                total_sys_time = 0
                for elem in elems_sys:
                    if elem[3] == i+1:
                        spkelem = []
                        spkelem.append(elem[1])
                        spkelem.append(elem[2])
                        spkelem.append(elem[3])
                        spkelems_sys_spk.append(spkelem)
                        total_sys_time += float(elem[2])
                spkelems_sys.append(spkelems_sys_spk)
                spktt_sys.append(total_sys_time)
            b = sorted(enumerate(spktt_sys), key=lambda x: x[1],reverse = True)
            spk_time_sys = []
            spk_elem_sys = []
            spk_id_sys = []
            for e in b:
                spk_time_sys.append(e[1])
                spk_elem_sys.append(spkelems_sys[e[0]])
                spk_id_sys.append(e[0]+1)
            fileid2segs_sys[fileid_sys] = spk_elem_sys
            fileid2time_sys[fileid_sys] = spk_time_sys
            file2spkid_sys[fileid_sys] = spk_id_sys

    fileid2segs_ref = {}
    fileid2time_ref = {}
    for (fileid_ref, elems_ref) in fileid2elems_ref.items():
        if fileid_ref not in fileid2elems_sys:
            print("seg pur error,there is no fileid_ref in sys rttm: %s" % fileid_ref)
        else:
            elems_ref = fileid2elems_ref[fileid_ref]
            spknum_ref = 0
            for elem in elems_ref:
                if elem[0] != fileid_ref:
                    printf("error, elem is not in fileid_sys")
                if elem[3] > spknum_ref:
                    spknum_ref = elem[3]
            file2spknum_ref[fileid_ref] = spknum_ref
            spkelems_ref = []
            spktt_ref = []
            for i in range(spknum_ref):
                spkelems_ref_spk = []
                total_ref_time = 0
                for elem in elems_ref:
                    if elem[3] == i + 1:
                        spkelem = []
                        spkelem.append(elem[1])
                        spkelem.append(elem[2])
                        spkelem.append(elem[3])
                        spkelems_ref_spk.append(spkelem)
                        total_ref_time += float(elem[2])
                spkelems_ref.append(spkelems_ref_spk)
                spktt_ref.append(total_ref_time)
            b = sorted(enumerate(spktt_ref), key=lambda x: x[1], reverse=True)
            spk_time_ref = []
            spk_elem_ref = []
            spk_id_ref = []
            for e in b:
                spk_time_ref.append(e[1])
                spk_elem_ref.append(spkelems_ref[e[0]])
                spk_id_ref.append(e[0] + 1)
            fileid2segs_ref[fileid_ref] = spk_elem_ref
            fileid2time_ref[fileid_ref] = spk_time_ref
            file2spkid_ref[fileid_ref] = spk_id_ref

    if len(fileid2segs_ref) != len(fileid2segs_sys):
        print("error, length of filedi2 segs_ref != fileid2segs_sys")
    else:
        file2cornum = {}
        file2totalnum = {}
        file2avetime = {}
        file2totaltime = {}
        file2errtime = {}
        for (fileid_sys, elems_sys) in fileid2segs_sys.items():
            spknum_ref = file2spknum_ref[fileid_sys]
            spknum_sys = file2spknum_sys[fileid_sys]
            ori_spkid_ref = file2spkid_ref[fileid_sys]
            ori_spkid_sys = file2spkid_sys[fileid_sys]
            elems_ref = fileid2segs_ref[fileid_sys]
            ref_idmap = []
            for i in range(spknum_ref):
                ref_idmap.append(1)
            spkid_cor_time = []
            spkid_total_time = []
            spkid_sys = []
            spkid_ref = []
            cor_num = 0
            tot_num = 0
            cor_time = 0
            tot_time = 0
            err_time = 0
            ave_time = 0
            cor_segnum = 0
            for ii in range(len(elems_sys)):
                elem_spk_sys = elems_sys[ii]
                for elem_ii in elem_spk_sys:
                    tot_num += 1
                    tot_time += elem_ii[1]
                    st_sys = elem_ii[0]
                    et_sys = elem_ii[0] + elem_ii[1]
                    id_sys = elem_ii[2]
                    ave_time += elem_ii[1]
                    last_spkid = -1
                    flag = 0
                    ana_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ana_time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    err_seg_time = 0
                    for i in range(len(elems_ref)):
                        elem_spk_ref = elems_ref[i]
                        for elem_i in elem_spk_ref:
                            st_ref = elem_i[0]
                            et_ref = elem_i[0] + elem_i[1]
                            id_ref = elem_i[2]
                            sysid_file = file2sysid[fileid_sys]
                            refid_file = file2refid[fileid_sys]
                            for temi in range(len(sysid_file)):
                                if sysid_file[temi]+1 == id_sys:
                                    id_ref_map = refid_file[temi]+1
                            if id_ref_map != id_ref:
                                if et_ref > st_sys and st_ref < st_sys:
                                    err_seg_time += min(et_sys, et_ref) - st_sys
                                else:
                                    if et_ref >= et_sys and st_ref < et_sys:
                                        err_seg_time += et_sys - max(st_sys, st_ref)
                                    else:
                                        if st_ref >= st_sys and et_ref <= et_sys:
                                            err_seg_time += et_ref - st_ref
                        
                        for elem_i in elem_spk_ref:
                            st_ref = elem_i[0]
                            et_ref = elem_i[0] + elem_i[1]
                            if st_ref-colar <= st_sys and et_ref+colar >= et_sys:
                                cor_num += 1
                                cor_time += et_sys - st_sys
                                flag = 1
                                break
                            else:
                                if st_sys <= st_ref and et_sys > st_ref:
                                    if last_spkid == -1:
                                        last_spkid = elem_i[2]
                                        ana_id[elem_i[2]] += 1
                                        id_time = min(et_sys,et_ref)
                                        ana_time[elem_i[2]] += id_time-st_ref
                                    else:
                                        if last_spkid != elem_i[2]:
                                            last_spkid = elem_i[2]
                                            ana_id[elem_i[2]] += 1
                                            id_time = min(et_sys, et_ref)
                                            ana_time[elem_i[2]] += id_time-st_ref
                                        else:
                                            ana_id[elem_i[2]] += 1
                                            id_time = min(et_sys, et_ref)
                                            ana_time[elem_i[2]] += id_time-st_ref
                                if st_sys <= et_ref and et_sys > et_ref:
                                    if last_spkid == -1:
                                        last_spkid = elem_i[2]
                                        ana_id[elem_i[2]] += 1
                                        id_time = max(st_sys, st_ref)
                                        ana_time[elem_i[2]] += et_ref - id_time
                                    else:
                                        if last_spkid != elem_i[2]:
                                            last_spkid = elem_i[2]
                                            ana_id[elem_i[2]] += 1
                                            id_time = max(st_sys, st_ref)
                                            ana_time[elem_i[2]] += et_ref - id_time
                                        else:
                                            ana_id[elem_i[2]] += 1
                                            id_time = max(st_sys, st_ref)
                                            ana_time[elem_i[2]] += et_ref - id_time
                                
                    if err_seg_time <= colar:
                        cor_segnum += 1

                    maxtime = -1
                    totaltime = 0
                    maxid = -1
                    for i in range(len(ana_time)):
                        spktime = ana_time[i]
                        totaltime += spktime
                        if spktime > maxtime:
                            maxtime = spktime
                            maxid = i
                    spk_er_time = totaltime-maxtime
                    if spk_er_time<= colar:
                        spk_er_time = 0
                    err_time += spk_er_time

                    if flag == 0 and spk_er_time == 0:
                        cor_num += 1
                        
            ave_time = ave_time / tot_num;
            file2avetime[fileid_sys] = ave_time
            file2cornum[fileid_sys] = cor_num
            file2totalnum[fileid_sys] = tot_num
            file2errtime[fileid_sys] = err_time
            file2totaltime[fileid_sys] = tot_time
            file2cor_segnum[fileid_sys] = cor_segnum

    return file2cornum,file2totalnum,file2errtime,file2totaltime,file2avetime,file2cor_segnum

def eval_elems_seg(fileid2elems_sys,fileid2elems_ref,colar,file2sysid,file2refid):
    file2cor_segtrans_num = {}
    file2cor_id_segtrans_num = {}
    file2sys_segtrans_num = {}
    file2ref_segtrans_num = {}
    file2sys_segtrans = {}
    file2ref_segtrans = {}
    for (fileid_sys, elems_sys) in fileid2elems_sys.items():
        if fileid_sys not in fileid2elems_ref:
            print("eval_elems_seg error,there is no fileid_sys in ref rttm: %s", fileid_sys)
        else:
            segtrans_elems_sys = []
            last_spkid_sys = elems_sys[0][3]
            last_end_sys = elems_sys[0][1] + elems_sys[0][2]
            for elem in elems_sys:
                if elem[0] != fileid_sys:
                    printf("error, elem is not in fileid_sys")
                if elem[3] != last_spkid_sys:
                    seg_elem_sys = []
                    seg_elem_sys.append(last_end_sys)
                    seg_elem_sys.append(last_spkid_sys)
                    seg_elem_sys.append(elem[3]);
                    seg_elem_sys.append(elem[1])
                    segtrans_elems_sys.append(seg_elem_sys)
                    last_spkid_sys = elem[3]
                last_end_sys = elem[1] + elem[2]
            file2sys_segtrans[fileid_sys] = segtrans_elems_sys
            file2sys_segtrans_num[fileid_sys] = len(segtrans_elems_sys)
    for (fileid_ref, elems_ref_1) in fileid2elems_ref.items():
        if fileid_sys not in fileid2elems_sys:
            print("eval_elems_seg error,there is no fileid_sys in sys rttm: %s", fileid_sys)
        else:
            elems_ref = sorted(enumerate(elems_ref_1), key=lambda x: x[1][1], reverse=False)
            segtrans_elems_ref = []
            last_spkid_ref = elems_ref[0][1][3]
            last_end_ref = elems_ref[0][1][1] +  elems_ref[0][1][2]
            for elem in elems_ref:
                if elem[1][0] != fileid_ref:
                    printf("error, elem is not in fileid_sys")
                if elem[1][3] != last_spkid_ref:
                    seg_elem_ref = []
                    seg_elem_ref.append(last_end_ref)
                    seg_elem_ref.append(last_spkid_ref);
                    seg_elem_ref.append(elem[1][3]);
                    seg_elem_ref.append(elem[1][1])
                    segtrans_elems_ref.append(seg_elem_ref)
                    last_spkid_ref = elem[1][3]
                last_end_ref = elem[1][1] + elem[1][2]
            file2ref_segtrans[fileid_ref] = segtrans_elems_ref
            file2ref_segtrans_num[fileid_ref] = len(segtrans_elems_ref)

    for (fileid_ref, segtrans_elems_ref) in file2ref_segtrans.items():
        cor_segtrans_num = 0
        cor_segtrans_id_num = 0
        sysid_file = file2sysid[fileid_ref]
        refid_file = file2refid[fileid_ref]
        for elem_ref in segtrans_elems_ref:
            beg_ref = float(elem_ref[0])
            end_ref = float(elem_ref[3])
            last_spkid_ref = elem_ref[1]
            spkid_ref = elem_ref[2]
            for elem_sys in file2sys_segtrans[fileid_ref]:
                beg_sys = float(elem_sys[0])
                end_sys = float(elem_sys[3])
                last_spkid_sys = elem_sys[1]
                spkid_sys = elem_sys[2]
                if beg_sys >= beg_ref - colar and end_sys <= end_ref + colar:
                    cor_segtrans_num += 1
                    for temi in range(len(sysid_file)):
                        if sysid_file[temi]+1 == last_spkid_sys and refid_file[temi]+1 == last_spkid_ref:
                            for temj in range(len(sysid_file)):
                                if sysid_file[temj]+1 == spkid_sys and refid_file[temj]+1 == spkid_ref:
                                    cor_segtrans_id_num += 1
        file2cor_segtrans_num[fileid_ref] = cor_segtrans_num
        file2cor_id_segtrans_num[fileid_ref] = cor_segtrans_id_num
    return file2cor_segtrans_num,file2cor_id_segtrans_num,file2sys_segtrans_num,file2ref_segtrans_num


def main():
    inreflist = sys.argv[1]
    sysfile = sys.argv[2]
    refile = sys.argv[3]
    colar = float(sys.argv[4])

    fileid2elems_ref = readlist_fileid2elems(inreflist)
    fileid2elems_sys = readfile_fileid2elems(sysfile)
    file2core_time, file2total_time, file2sysid, file2refid, file2spknum_sys, file2spknum_ref = eval_elems(fileid2elems_sys,fileid2elems_ref)
    file2cornum,file2totalnum,file2errtime,file2totaltime,file2avetime,file2cor_segnum = eval_elems_pur(fileid2elems_sys,fileid2elems_ref,colar,file2sysid,file2refid)
    file2cor_segtrans_num,file2cor_id_segtrans_num,file2sys_segtrans_num,file2ref_segtrans_num = eval_elems_seg(fileid2elems_sys,fileid2elems_ref,colar,file2sysid,file2refid)
    all_cor_time = 0
    all_time = 0
    err_spknum = 0
    total_spknum_ref = 0
    cornum_total = 0
    totnum_total = 0
    errtime_total = 0
    tottime_total = 0
    total_cor_segtrans_num = 0
    total_cor_id_segtrans_num = 0
    total_sys_segtrans_num = 0
    total_ref_segtrans_num = 0
    total_cor_segnum = 0
    with open(refile,"w") as fop:
        for (fileid,cor_time_file) in file2core_time.items():
            fop.write("%s\n" % fileid)
            total_time_file = file2total_time[fileid]
            file_cor_time = 0
            file_time = 0
            spkid_sys = file2sysid[fileid]
            for i in range(len(cor_time_file)):
                file_cor_time += cor_time_file[i]
                file_time += total_time_file[i]
                if total_time_file[i] == 0:
                    cor_rate_spk = 0
                else:
                    cor_rate_spk = cor_time_file[i] * 100 / total_time_file[i]
                fop.write("spkid %d total time: %f\t correct time: %f\t correct rate: %f %%\n" % (spkid_sys[i],total_time_file[i],cor_time_file[i],cor_rate_spk))
            if file_time == 0:
                cor_rate = 0
            else:
                cor_rate = file_cor_time*100/file_time
            all_cor_time += file_cor_time
            all_time += file_time
            spknumsys = file2spknum_sys[fileid]
            spknumref = file2spknum_ref[fileid]
            err_spknum += abs(spknumsys - spknumref)
            total_spknum_ref += spknumref
            sysid_file = file2sysid[fileid]
            refid_file = file2refid[fileid]
            fop.write(" ref spk number: %d\t sys spk number: %d\n" % (spknumref,spknumsys))
            fop.write(" total time: %f\t correct time: %f\t correct rate: %f %%\n" % (file_time,file_cor_time,cor_rate))
            if len(sysid_file) != len(refid_file):
                print("length of ref spkid != sys spkid\n")
            else:
                for i in range(len(refid_file)):
                    fop.write("%d -> %d\t" %(refid_file[i],sysid_file[i]))
                fop.write("\n")

            cornum = file2cornum[fileid]
            totnum = file2totalnum[fileid]
            cornum_total += cornum
            totnum_total += totnum
            if totnum == 0:
                pur_num_rate = 0
            else:
                pur_num_rate = cornum * 100 / totnum
                prec_seg = file2cor_segnum[fileid] * 100 / totnum
            errtime = file2errtime[fileid]
            tottime = file2totaltime[fileid]
            errtime_total += errtime
            tottime_total += tottime
            if tottime == 0:
                err_time_rate = 0
            else:
                err_time_rate = errtime * 100 / tottime
            fop.write(
                " pur num: %d\t cor id trans num: %d\t tot num: %d\t pur rate:%f %%  err time: %f\t total time:%f\t err_time_rate:%f %%\t prec_seg: %f %%\t ave_time:%f\n" % (
                cornum, file2cor_segnum[fileid], totnum, pur_num_rate, errtime, tottime, err_time_rate, prec_seg, file2avetime[fileid]))
            total_cor_segnum += file2cor_segnum[fileid]
            if file2ref_segtrans_num[fileid] == 0:
                recall = 0
                recall_id = 0
            else:
                recall = file2cor_segtrans_num[fileid] * 100 / file2ref_segtrans_num[fileid]
                recall_id = file2cor_id_segtrans_num[fileid] * 100 / file2ref_segtrans_num[fileid]
            if file2sys_segtrans_num[fileid] == 0:
                prec = 0
                prec_id = 0
            else:
                prec = file2cor_segtrans_num[fileid] * 100 / file2sys_segtrans_num[fileid]
                prec_id = file2cor_id_segtrans_num[fileid] * 100 / file2sys_segtrans_num[fileid]

            fop.write(
                "ref trans num: %d\t sys trans num: %d\t cor trans num: %d\t cor trans id num: %d\t recall:%f %%\t prec:%f %%\t recall_id:%f %%\t prec_id:%f %%\n" % (
                file2ref_segtrans_num[fileid], file2sys_segtrans_num[fileid], file2cor_segtrans_num[fileid],file2cor_id_segtrans_num[fileid], recall, prec,recall_id, prec_id))

            total_cor_segtrans_num += file2cor_segtrans_num[fileid]
            total_cor_id_segtrans_num += file2cor_id_segtrans_num[fileid]
            total_sys_segtrans_num += file2sys_segtrans_num[fileid]
            total_ref_segtrans_num += file2ref_segtrans_num[fileid]

        fop.write("*******************Total result:******************************\n")
        if all_time == 0:
            cor_rate = 0
        else:
            cor_rate = all_cor_time*100/all_time
        fop.write("total time: %f\t correct time: %f\t correct rate: %f %% ref_spk_num: %d err_spk_num: %d\n" % (all_time,all_cor_time,cor_rate, total_spknum_ref, err_spknum))

        pur_num_rate_total = 0
        err_time_rate_total = 0
        if totnum_total == 0:
            pur_num_rate_total = 0
        else:
            pur_num_rate_total = cornum_total*100/totnum_total
            prec_seg_total = total_cor_segnum*100/totnum_total
        if tottime_total == 0:
            err_time_rate_total = 0
        else:
            err_time_rate_total = errtime_total*100/tottime_total
        fop.write("total pur num: %d\t cor id seg num: %d\t tot num: %d\t pur rate:%f %%  err time: %f\t total time:%f\t err_time_rate:%f %%\t seg prec rate:%f %% \n" % (cornum_total,total_cor_segnum,totnum_total,pur_num_rate_total,errtime_total,tottime_total,err_time_rate_total,prec_seg_total))

        if total_ref_segtrans_num == 0:
            recall = 0
            recall_id = 0
        else:
            recall = total_cor_segtrans_num*100/total_ref_segtrans_num
            recall_id = total_cor_id_segtrans_num*100/total_ref_segtrans_num
        if total_sys_segtrans_num == 0:
            prec = 0
            prec_id = 0
        else:
            prec = total_cor_segtrans_num*100/total_sys_segtrans_num
            prec_id = total_cor_id_segtrans_num*100/total_sys_segtrans_num
        
        fop.write("total ref trans num: %d\t total sys trans num: %d\t total cor trans num: %d\t total cor trans id num: %d\t recall:%f %%\t prec:%f %%\t recall_id:%f %%\t prec_id:%f %%\n" % (total_ref_segtrans_num,total_sys_segtrans_num,total_cor_segtrans_num,total_cor_id_segtrans_num,recall,prec,recall_id,prec_id))


if __name__ == "__main__":
    main()
