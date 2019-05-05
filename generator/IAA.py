
from sklearn.metrics import cohen_kappa_score
import os
from collections import defaultdict

class Agreement():
    """
    compute Cohen's kappa agreement between annotators
    
    """
    
    def __init__(self):
        self.annotator1 = defaultdict(dict)
        self.annotator2 = defaultdict(dict)
        self.annotator3 = defaultdict(dict)


    def get_data(self, file, coder):
        
        with open(file) as f:
            context = f.read()
            
        lines = context.strip().split('\n\n')
        for line in lines:
            data = line.split('|')
            annotation_id = data[0]
            event_text = data[2]
            event_confidence = data[3]
            source = data[4]
            emotion = data[5]
            emotion_value = data[6]
            
            if coder == 'annotator1':
                self.annotator1[annotation_id][event_text] = [event_confidence, source, emotion, emotion_value]
            elif coder == 'annotator2':
                self.annotator2[annotation_id][event_text] = [event_confidence, source, emotion, emotion_value]
            else:
                self.annotator3[annotation_id][event_text] = [event_confidence, source, emotion, emotion_value]
            
            
    def get_agreement(self):
        
        #3 annotators
        file_dirs = ['data/annotator1', 'data/annotator2', 'data/annotator3']
        for file_dir in file_dirs:
            file_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', file_dir))
            for root, dirs, files in os.walk(file_dir):
                files = [os.path.join(root, name) for name in files if name[-3:] =='txt']
                coder = file_dir.split('/')[-1]
                for file in files:
                    self.get_data(file, coder)
        
        #agreement between annotator 1 and 2 
        anno_1_2 = [[],[]]      
        for annotation_id in self.annotator1:
            annotator2 = self.annotator2[annotation_id]
            if annotator2:
                for event, value in self.annotator1[annotation_id].items():
                    #anno_1_2[0].extend(['annotator1', event, value[0], value[1], value[2], value[3]])
                    anno_1_2[0].extend(['annotator1',  event, value[2], value[3]])
                    if event in self.annotator2[annotation_id]:
                        #anno_1_2[1].extend(['annotator2', event, self.annotator2[annotation_id][event][0],self.annotator2[annotation_id][event][1], self.annotator2[annotation_id][event][2],self.annotator2[annotation_id][event][3]])
                        anno_1_2[1].extend(['annotator2', event, self.annotator2[annotation_id][event][2], self.annotator2[annotation_id][event][3]])
                    else:
                        #anno_1_2[1].extend(['annotator2', '0', '0', '0','0','0'])
                        anno_1_2[1].extend(['annotator2', '0','0', '0'])
        print("Cohen's kappa score between annotator 1 and 2: \n", cohen_kappa_score(anno_1_2[0], anno_1_2[1]))
       
        #agreement between annotator 1 and 3 
        anno_1_3 = [[],[]]      
        for annotation_id in self.annotator1:
            annotator3 = self.annotator3[annotation_id]
            if annotator3:
                for event, value in self.annotator1[annotation_id].items():
                    #anno_1_3[0].extend(['annotator1', event, value[0], value[1], value[2], value[3]])
                    anno_1_3[0].extend(['annotator1', event, value[2], value[3]])
                    if event in self.annotator3[annotation_id]:
                        #anno_1_3[1].extend(['annotator3', event, self.annotator3[annotation_id][event][0],self.annotator3[annotation_id][event][1], self.annotator3[annotation_id][event][2],self.annotator3[annotation_id][event][3]])
                        anno_1_3[1].extend(['annotator3', event, self.annotator3[annotation_id][event][2], self.annotator3[annotation_id][event][3]])
                    else:
                        #anno_1_3[1].extend(['annotator3', '0', '0', '0','0','0'])
                        anno_1_3[1].extend(['annotator3', '0', '0','0'])
        print("Cohen's kappa score between annotator 1 and 3: \n", cohen_kappa_score(anno_1_3[0], anno_1_3[1]))
       
        #agreement between annotator 2 and 3 
        anno_2_3 = [[],[]]      
        for annotation_id in self.annotator2:
            annotator3 = self.annotator3[annotation_id]
            if annotator3:
                for event, value in self.annotator2[annotation_id].items():
                    #anno_2_3[0].extend(['annotator2', event, value[0], value[1], value[2], value[3]])
                    anno_2_3[0].extend(['annotator2', event, value[2], value[3]])
                    if event in self.annotator3[annotation_id]:
                        #anno_2_3[1].extend(['annotator3', event, self.annotator3[annotation_id][event][0],self.annotator3[annotation_id][event][1],self.annotator3[annotation_id][event][2],self.annotator3[annotation_id][event][3]])
                        anno_2_3[1].extend(['annotator3', event, self.annotator3[annotation_id][event][2], self.annotator3[annotation_id][event][3]])
                    else:
                        #anno_2_3[1].extend(['annotator3', '0', '0', '0','0','0'])
                        anno_2_3[1].extend(['annotator3', '0', '0','0'])
        print("Cohen's kappa score between annotator 2 and 3: \n", cohen_kappa_score(anno_2_3[0], anno_2_3[1]))
       
                    
if __name__ == '__main__':
    iaa = Agreement()
    iaa.get_agreement()                    
    

        