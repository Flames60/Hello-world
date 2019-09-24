from __future__ import print_function, absolute_import
import os.path as osp
import os
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
import numpy as np

class hy1kdomain_mars(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(self.__class__, self).__init__(root, split_id=split_id)
        self.name="hy1kdomain_mars"
        self.num_cams = 50
        self.is_video = False
        self.source_test = '/home/gongweibo/tip_code/data/hy4000_test'
        self.target_test = '/home/gongweibo/tip_code/data/mars/test'
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")
        self.load_2do(num_val)
    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        print("create new dataset")
        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities = [[{} for _ in range(self.num_cams)] for _ in range(100001)]
     #   identities = [[] for _ in range(1000)]
        video_id = 0
        
        id2index = {}
        def cam_stats(subdir):
            cams=[]

            # subdir = osp.join(self.target_test, subdir)#'/home/ligungrui/one-Example-Person-ReID-master/data/hy4000_test/gallery'#os.path.join(self.root, subdir)
            for d in os.listdir(subdir):
                tmp_ls = os.listdir(os.path.join(subdir,d))
#                print(subdir.split("/")[-2])
                if subdir.split("/")[-2] == "mars" or subdir.split("/")[-3] == "mars":
                    tmp_cams = [int(i[5]) for i in tmp_ls]
                  
                else:
                    tmp_cams = [int(i.split('_')[4]) for i in tmp_ls]
                cams.extend(tmp_cams)
            unique = np.unique(cams)
            cam_dic = {}
            for ind,u in enumerate(unique):
                cam_dic[u]=ind
            print('{} cameras totally'.format(len(cam_dic)))
            return cam_dic

        def register(subdir, cam_dic, video_id, source_path):
            pids = set()
            relabeled_pid = -1
            vids = []
            source_path = source_path
            
            person_list = os.listdir(os.path.join(source_path, subdir)); person_list.sort()
#            print(len(person_list))
#            exit()

            for index, person_id in enumerate(person_list):
                count = 0
                if person_id not in id2index:
                    id2index[person_id]= len(id2index)
                pid = id2index[person_id]
                videos = os.listdir(os.path.join(source_path, subdir, person_id)); videos.sort()
                list_bycam = {cam:[] for cam in cam_dic.keys()}
                if pid<0:
                    continue
                for vid in videos:
                    # if source_path.split("/")[-1] != "mars":
                    if subdir== "gallery" or subdir == "query":
                        if source_path.split("/")[-2] != "mars":
                            cu_cam = int(vid.split('_')[4])
                        else:
                            cu_cam = int(vid[5])
                            if cu_cam not in [1,2,3,4,5,6]:
                                continue

                    else:
                        #cu_cam = int(vid[5])
                        #if cu_cam not in [1,2,3,4,5,6]:
                        #    continue
                        cu_cam = int(vid.split('_')[4])
                    list_bycam[cu_cam].append(vid)
                for cam in list_bycam.keys():
                    video_path = os.path.join(source_path, subdir, person_id)
                    tmp_ls = list_bycam[cam]
                    if len(tmp_ls)==0:
                        continue

                    image_path = os.path.join(self.root,subdir)                    
                    if "mars" in  image_path.split("/")[-2]:

                        tracklet_dict = {}
                        for im_file in tmp_ls:
                            tracklet_id = im_file[6:11]
                            if tracklet_id not in tracklet_dict.keys():
                                tracklet_dict[tracklet_id]=[im_file]
                            else:
                                tracklet_dict[tracklet_id].append(im_file)

                        # frame_list = []

                        for tracklet_id in tracklet_dict.keys():
                            cu_vid = video_id#current video id
                            frame_list = []
                            # print("cu_vid", cu_vid)
                            for index, item in enumerate(tracklet_dict[tracklet_id]):
                                newname = ('{:04d}_{:02d}_{:05d}_{:04d}.jpg'.format(pid, cam_dic[cam], cu_vid, len(frame_list)))
                                print("newname", newname)
                                pids.add(pid)
                   #         print(newname)
                                frame_list.append(newname)
                                shutil.copyfile(osp.join(video_path, item), osp.join(images_dir, newname))
                            print("video_id", video_id)
                            video_id+=1
                    #tmp_dict = {cu_vid:frame_list}
                            print(pid, len(cam_dic), cam, video_id)
                            identities[pid][cam_dic[cam]][video_id] = frame_list                
                    #identities[pid][cam_dic[cam]]=tmp_dict
                            vids.append(frame_list)

                            print("ID {}, frames {}\t  in {} {}".format(pid, len(frame_list), video_id, subdir))

                    else:
                        cu_vid = video_id#current video id
                        frame_list = []
                        for index, item in enumerate(tmp_ls):
                            newname = ('{:04d}_{:02d}_{:05d}_{:04d}.jpg'.format(pid, cam_dic[cam], cu_vid, len(frame_list)))
                            pids.add(pid)
                   #         print(newname)
                            frame_list.append(newname)
                            shutil.copyfile(osp.join(video_path, item), osp.join(images_dir, newname))
                        print("video_id", video_id)
                        video_id+=1
                        #tmp_dict = {cu_vid:frame_list}
                        print(pid, len(cam_dic), cam, video_id)
                        identities[pid][cam_dic[cam]][video_id] = frame_list                
                        #identities[pid][cam_dic[cam]]=tmp_dict
                        vids.append(frame_list)

                        print("ID {}, frames {}\t  in {} {}".format(pid, len(frame_list), video_id, subdir))    

            return pids, vids, video_id


        print("begin to preprocess " + self.name + " dataset")
#        cam_dic = cam_stats('/home/jovyan/cache/majian/codes/tip/one_example/data/mars/train_all_me')
        cam_dic = cam_stats("/home/gongweibo/tip_code/data/hy1kdomain_mars/train_all")
        trainval_pids, _, video_id = register('train_all', cam_dic, video_id, self.root)
        #Source domain testset
        # cam_dic = cam_stats('/home/wukebin/PytorchReIDCode/codes_majian/tip/one-Example-Person-ReID-master/data/mars')
        cam_dic = cam_stats("/home/gongweibo/tip_code/data/mars/test/gallery")
        t_gallery_pids, t_gallery_vids, video_id = register('gallery', cam_dic, video_id, self.target_test)
        t_query_pids, t_query_vids, video_id = register('query', cam_dic, video_id, self.target_test)
        #Target domain testset
        cam_dic = cam_stats('/home/gongweibo/tip_code/data/hy4000_test/gallery')
        s_gallery_pids, s_gallery_vids, video_id = register('gallery', cam_dic, video_id, self.source_test)
        s_query_pids, s_query_vids, video_id = register('query', cam_dic, video_id, self.source_test)

        #assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(t_gallery_pids)
       # print(list(trainval_pids)[0])
        # Save meta information into a json file
        meta = {'name': self.name, 'shot': 'multiple', 'num_cameras': self.num_cams,
                'identities': identities,
                'query': t_query_vids,
                'gallery': t_gallery_vids,
                's_query': s_query_vids,
                's_gallery': s_gallery_vids}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        
        splits = [{
            'train': sorted(list(trainval_pids)),
            'query': sorted(list(t_query_pids)) ,
            'gallery': sorted(list(t_gallery_pids)),
            's_query': sorted(list(s_query_pids)) ,
            's_gallery': sorted(list(s_gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))


