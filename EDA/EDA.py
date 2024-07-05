import pandas as pd
import os
import pandas_profiling
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
import csv
import matplotlib.pyplot as plt
import pdfkit
from pypdf import PdfWriter
import img2pdf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.manifold import TSNE
import librosa
from pandas_profiling import ProfileReport



def html_to_pdf(input_path, output_path):
    pdfkit.from_file(input_path, output_path)


## 13 MFCC FEATURES OF ANY RANDOM AUDIO, CAN ADD IN REPORT ACCODRINGLY
# Load audio file and extract MFCC features
audio_file = "/home/hiddenmist/Aman_Lakshay/VOXCELEB/EVALUATING_AUDIOS/CLASSES/id10270/5r0dWxy17C8/00003.wav"
y, sr = librosa.load(audio_file)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Create DataFrame with MFCC features
mfcc_df = pd.DataFrame(mfccs.T, columns=[f'MFCC_{i}' for i in range(1, 14)])

# Generate Pandas Profiling report
profile = pandas_profiling.ProfileReport(mfcc_df)
profile.to_file("mfcc_profile_report.html")
input_file = "mfcc_profile_report.html"
output_file = "mfcc_features.pdf"
html_to_pdf(input_file, output_file)


#Loading CSV File

directory = '/home/hiddenmist/Aman_Lakshay/EDA/free-spoken-digit-dataset-master/recordings'
path_to_csv='/home/hiddenmist/Aman_Lakshay/EDA/temp_audio.csv'
data = []

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        class_name = filename.split('_')[0]
        file_path = os.path.join(directory, filename)
        # print(file_path)
        data.append((file_path, class_name))

csv_filename = path_to_csv
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'label'])
    writer.writerows(data)

print(f"CSV file '{csv_filename}' created successfully.")


#TODO: Add this class to the existing code

class AudioFeaturesExtractor:
    def __init__(self, audio_path, label):
        self.audio_path = audio_path
        self.label = label
        self.audio_data, self.sample_rate = librosa.load(audio_path)
        self.pitch = self.pitch_ex(audio_path)
        self.rms_energy = self.compute_rms_energy()
        self.spectral_centroids = self.compute_spectral_centroids()
        self.audio_length = len(self.audio_data)
        self.mean = self.audio_data.mean()
        self.std = self.audio_data.std()
        

    def pitch_ex(self,audio_file):
        audio_data, sample_rate = librosa.load(audio_file)

        pitch, _ = librosa.core.piptrack(y=audio_data, sr=sample_rate)

        pitch = pitch[np.argmax(np.sum(pitch, axis=1))]
        return pitch
    

    def compute_rms_energy(self):
        rms_energy = librosa.feature.rms(y=self.audio_data)
        return rms_energy

    def compute_spectral_centroids(self):
        spectral_centroids = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate)
        return spectral_centroids

    def length(self):
        return self.audio_length

    def get_item(self, item):
        if item == 'audio_path':
            return self.audio_path
        elif item == 'class':
            return self.label
        elif item == 'audio_length':
            return self.audio_length
        elif item == 'mean':
            return self.mean
        elif item == 'std':
            return self.std
        elif item == 'sample_rate':
            return self.sample_rate
        elif item == 'pitch':
            return self.pitch
        elif item == 'rms_energy':
            return self.rms_energy
        elif item == 'spectral_centroids':
            return self.spectral_centroids

    def to_dict(self):
        return {
            'audio_path': self.audio_path,
            'class': self.label,
            'audio_length': self.audio_length,
            'mean': self.mean,
            'std': self.std,
            'sample_rate': self.sample_rate,
            'pitch': self.pitch,
            'rms_energy': self.rms_energy,
            'spectral_centroids': self.spectral_centroids
        }
        
#Loading DataFrame
      
df = pd.read_csv(path_to_csv) 
audio_data = []
features_list=[]
audio_label=[]
a=0
a_count=0
for index, row in df.iterrows():
    # pass
    file_path = row['path']
    label=row['label']
    audio_label.append(label)
    audio, sample_rate = librosa.load(file_path)
    extractor = AudioFeaturesExtractor(file_path, label)
    
    features = extractor.to_dict()
    # mfcc_feat=cepstral_feat(file_path)
    
    features_list.append(features)
    
    a+=len(audio)
    a_count+=1
    audio_data.append((audio, sample_rate))

# Pandas Profiling part of report
df = pd.DataFrame(features_list)
profile_audio = ProfileReport(df, title='Audio Data EDA')
profile_audio.to_file("audio_data_eda.html")



input_file = "audio_data_eda.html"
output_file = "output.pdf"
output_file2="audio_eda.pdf"
html_to_pdf(input_file, output_file)
html_to_pdf(input_file, output_file2)





class classFeaturesExtractor:
    def __init__(self, audio_data, audio_label):
        self.audio_map = defaultdict(list)
        for (audio, _), j in zip(audio_data, audio_label):
            self.audio_map[j].append(audio)
            
        self.mymap = defaultdict(dict)
        for i in audio_label:
            self.mymap[i]['class_count'] = 0

        for i in audio_label:
            self.mymap[i]['class_count'] += 1

        self.class_names = list(self.mymap.keys())
        self.class_counts = [value['class_count'] for value in self.mymap.values()]

    def calculate_quartiles(self):
        self.first_quartile = np.percentile(self.class_counts, 25)
        self.median = np.percentile(self.class_counts, 50) 
        self.third_quartile = np.percentile(self.class_counts, 75)

    def determine_bal_state(self):
        for class_name, count in self.mymap.items():
            if count['class_count'] < self.first_quartile:
                print(f"Class '{class_name}' has count {count['class_count']}, which means it is a MIN OUTLIER class (less).")
                self.mymap[class_name]['bal_state (-1 means unbal_low, 0 means bal, 1 means unbal_high']=-1

            elif count['class_count'] > self.third_quartile:
                print(f"Class '{class_name}' has count {count['class_count']}, which means it is a MAX OUTLIER class (more).")
                self.mymap[class_name]['bal_state (-1 means unbal_low, 0 means bal, 1 means unbal_high']=1

            else:
                self.mymap[class_name]['bal_state (-1 means unbal_low, 0 means bal, 1 means unbal_high']=0

    def calculate_class_stats(self, audio_data, audio_label):
        class_mean = defaultdict(float)
        class_std = defaultdict(float)

        for (i, _), j in zip(audio_data, audio_label):
            c = np.mean(i)
            d = np.std(i)
            class_mean[j] += c
            class_std[j] += d

        for label in class_mean:
            count = self.mymap[label]['class_count']
            class_mean[label] /= count
            class_std[label] /= count

        for label in class_mean:
            self.mymap[label]['class_mean'] = class_mean[label]
            self.mymap[label]['class_std'] = class_std[label]

    def get_class_fea(self, label):
        count = self.mymap[label]['class_count']
        bal_state = self.mymap[label]['bal_state (-1 means unbal_low, 0 means bal, 1 means unbal_high']
        class_mean = self.mymap[label]['class_mean']
        class_std = self.mymap[label]['class_std']

        return {
            'class': label,
            'class_count': count,
            'bal_state': bal_state,
            'class_mean': class_mean,
            'class_std': class_std
            # Add more features as needed
        }

    def create_class_features_dataframe(self):
        class_features = []
        for i in self.class_names:
            class_feat = self.get_class_fea(i)
            class_features.append(class_feat)
        df_class = pd.DataFrame(class_features)
        return df_class

df=pd.read_csv(path_to_csv)
audio_data = []
features_list=[]
audio_label=[]
a=0
a_count=0
for index, row in df.iterrows():
    # pass
    file_path = row['path']
    label=row['label']
    audio_label.append(label)
    audio, sample_rate = librosa.load(file_path)
    features_list.append(features)
    a+=len(audio)
    a_count+=1
    audio_data.append((audio, sample_rate))

class_processor = classFeaturesExtractor(audio_data, audio_label)
class_processor.calculate_quartiles()
class_processor.determine_bal_state()
class_processor.calculate_class_stats(audio_data, audio_label)
df_class = class_processor.create_class_features_dataframe()

profile_audio = ProfileReport(df, title='Class Data EDA')
profile_audio.to_file("class_data_eda.html")


input_file = "class_data_eda.html"
output_file = "output1.pdf"
html_to_pdf(input_file, output_file)

class EDA_items():
    def __init__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        self.df = None
        self.audio_data = []
        self.audio_label = []
        self.audio_map = defaultdict(list)
        self.mymap = defaultdict(int)
        self.avg_a = 0
        self.class_mean = {}
        self.class_std = {}
    def load_data(self):
        self.df = pd.read_csv(self.path_to_csv)
        
        
    def process_audio_data(self):
        a = 0
        a_count = 0
        
        for index, row in self.df.iterrows():
            file_path = row['path']
            label = row['label']
            self.audio_label.append(label)
            audio, sample_rate = librosa.load(file_path)
            a += len(audio)
            a_count += 1
            self.audio_data.append((audio, sample_rate))
            self.avg_a = int(a / a_count)

        for idx, (audio, _) in enumerate(self.audio_data):
            arr = np.array(audio)
            diff = len(arr) - self.avg_a
            if diff < 0:  # padding
                pad_before = abs(diff) // 2
                pad_after = abs(diff) - pad_before
                arr_modified = np.pad(arr, (pad_before, pad_after), mode='constant')
            elif diff > 0:  # cropping
                crop_start = diff // 2
                crop_end = crop_start + self.avg_a
                arr_modified = arr[crop_start:crop_end]
            else:  # equal
                arr_modified = arr

            self.audio_data[idx] = (arr_modified, _)

        for (audio, _), j in zip(self.audio_data, self.audio_label):
            self.audio_map[j].append(audio)

        for i in self.audio_label:
            self.mymap[i] += 1
            
    def plot_class_counts(self):
        # Assuming mymap contains the counts for each class label
        class_labels = list(self.mymap.keys())
        class_counts = list(self.mymap.values())

        # Plot the bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, class_counts, color='skyblue')
        plt.xlabel('Class Labels')
        plt.ylabel('Counts')
        plt.title('Counts of Audio Classes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Set x-axis ticks at every label
        plt.xticks(range(len(class_labels)), class_labels)
        plt.savefig('check_balancing.png')
        
        
    def quartile_class_count(self):
        class_counts = list(self.mymap.values())

        first_quartile = np.percentile(class_counts, 25)
        median = np.percentile(class_counts, 50)  
        third_quartile = np.percentile(class_counts, 75)

        output_pdf_path = "quartile_analysis.pdf"

        # Create a PDF file to save the output
        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        y = 750  # Starting y-coordinate

        c.drawString(50, y, "First Quartile: {}".format(first_quartile))
        y -= 20
        c.drawString(50, y, "Median: {}".format(median))
        y -= 20
        c.drawString(50, y, "Third Quartile: {}".format(third_quartile))
        y -= 20
        c.drawString(50, y, "Class Counts Analysis:")
        y -= 20

        for class_name, count in self.mymap.items():
            if count < first_quartile:
                text = f"Class '{class_name}' has count {count}, which means it is a MIN OUTLIER class (less)."
            elif count > third_quartile:
                text = f"Class '{class_name}' has count {count}, which means it is a MAX OUTLIER class (more)."
            else:
                text = f"Class '{class_name}' has count {count}, which is within the quartile range."
            
            # Write to PDF
            c.drawString(50, y, text)
            y -= 20

        # Save the PDF
        c.save()
    
    
    def mean_std_class(self):
        self.class_mean = defaultdict(float)
        self.class_std = defaultdict(float)

        for (i,_),j in zip(self.audio_data, self.audio_label):
            c = np.mean(i)
            d = np.std(i)
            self.class_mean[j] += c
            self.class_std[j] += d

        for label in self.class_mean:
            count = self.mymap[label]
            self.class_mean[label] /= count
            self.class_std[label] /= count

        output_pdf_path = "mean_std_analysis.pdf"

        # Create a PDF file to save the output
        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        y = 750  # Starting y-coordinate

        c.drawString(50, y, "Mean and Standard Deviation Analysis:")
        y -= 20

        for label in self.class_mean:
            mean = self.class_mean[label]
            std = self.class_std[label]
            text = f"Class {label}: Mean = {mean}, Std = {std}"
            
            # Write to PDF
            c.drawString(50, y, text)
            y -= 20

        # Save the PDF
        c.save()

    def plot_mean_quartile(self):
        class_names = list(self.class_mean.keys())
        class_means = list(self.class_mean.values())

        first_quartile = np.percentile(class_means, 25)
        median = np.percentile(class_means, 50)  
        third_quartile = np.percentile(class_means, 75)

        plt.figure(figsize=(8, 6))
        plt.boxplot(class_means, vert=False, widths=0.7, patch_artist=True)
        plt.scatter([first_quartile, median, third_quartile], [1, 1, 1], color='red', zorder=5)
        plt.xlabel('Mean of Audio Data')
        plt.title('Quartiles of Mean Audio Data')
        plt.yticks([])
        plt.grid(True)

        for i, (mean, label) in enumerate(zip(class_means, class_names)):
            plt.text(mean, 1, f'{label}', ha='center', va='center', fontsize=8)

        output_pdf_path = "mean_quartile_analysis.pdf"
        plt.savefig(output_pdf_path)
        
        # print("Plot saved as:", output_pdf_path)

    def plot_std_quartile(self, output_pdf_path="std_quartile_analysis.pdf"):
        class_names = list(self.class_std.keys())
        class_stds = list(self.class_std.values())

        first_quartile = np.percentile(class_stds, 25)
        median = np.percentile(class_stds, 50)
        third_quartile = np.percentile(class_stds, 75)

        plt.figure(figsize=(8, 6))
        plt.boxplot(class_stds, vert=False, widths=0.7, patch_artist=True)
        plt.scatter([first_quartile, median, third_quartile], [1, 1, 1], color='red', zorder=5)
        plt.xlabel('Standard Deviation of Audio Data')
        plt.title('Quartiles of Standard Deviation Audio Data')
        plt.yticks([])
        plt.grid(True)

        for i, (std, label) in enumerate(zip(class_stds, class_names)):
            plt.text(std, 1, f'{label}', ha='center', va='center', fontsize=8)

        plt.savefig(output_pdf_path)
       
    def make_pdf(self,pdfs):
        merger = PdfWriter()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write("output.pdf")
        merger.close()

    def cos_sim(self, aud1, aud2):
        audio_1 = aud1.reshape(1, -1)
        audio_2 = aud2.reshape(1, -1)
        return cosine_similarity(audio_1, audio_2)[0][0]

    def print_cosine_similarity_matrix(self, output_pdf_path=None):
        sim_list = []

        for class_label, audio_list in self.audio_map.items():
            random.shuffle(audio_list)
            selected_audios = audio_list[:10]  # 10 is the number of samples here
            for audio_array in selected_audios:
                sim_list.append((audio_array, class_label))

        # Calculate cosine similarity scores
        num_samples = len(sim_list)
        cosine_sim_scores = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                cosine_sim_scores[i, j] = self.cos_sim(sim_list[i][0], sim_list[j][0])

        # Generate default output PDF file name if not provided
        if output_pdf_path is None:
            output_pdf_path = 'cosine_similarity_output.pdf'

        # Write cosine similarity scores to a new PDF file
        with PdfPages(output_pdf_path) as pdf:
            for i in range(num_samples):
                for j in range(num_samples):
                    similarity_score = cosine_sim_scores[i, j]
                    pdf.savefig()
                    with open(output_pdf_path.replace('.pdf', '.txt'), 'a') as f:
                        f.write(f"Cosine similarity between sample {i+1} and sample {j+1}: {similarity_score}\n")
            plt.close()  # Move plt.close() outside of the inner loop


    def plot_gr(self, matri, title1,filename):
        matrix_values = np.array([[element[0][0] for element in row] for row in matri])
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix_values, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(title1)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(filename)

    def cos_sim(self,aud1,aud2):
    

        audio_1 = aud1.reshape(1, -1)
        audio_2 = aud2.reshape(1, -1)
        c = cosine_similarity(audio_1, audio_2)
        return c    
    
    def avg_similarity_matrix(self,avg_sim_list):
        avg_adjacency_matrix=[]
        
        for i in avg_sim_list:
            temp=[]
            for j in avg_sim_list:
                temp.append(self.cos_sim(i[0],j[0]))
            avg_adjacency_matrix.append(temp)
        return avg_adjacency_matrix
    
    def avg_adjacency_plot(self):
        avg_sim_list=[]
        for class_label, audio_list in self.audio_map.items():

            random.shuffle(audio_list)
            
            selected_audios = audio_list[:4] # 4 is the samples here
            average_audio = np.mean(selected_audios, axis=0)
            average_audio /= np.max(np.abs(average_audio))
            avg_sim_list.append((average_audio,class_label))
        avg_adjacency_matrix=self.avg_similarity_matrix(avg_sim_list)
        self.plot_gr(avg_adjacency_matrix,"Heatmap of Class_average_adjacency_matrix",'heatmap1.pdf')
        
    def cosine_similarity_matrix(self,sim_list):
        adjacency_matrix=[]
        
        for i in sim_list:
            temp=[]
            for j in sim_list:
                temp.append(self.cos_sim(i[0],j[0]))
            adjacency_matrix.append(temp)
        return adjacency_matrix
    
    def occurence_sim_plot(self):
        total_indices = len(audio_data)
        mini_batch = 32
        random_indices = np.random.choice(total_indices, mini_batch, replace=False)
        mini_batch_list=[]
        for i in random_indices:
            mini_batch_list.append(audio_data[i])
        occu_sim=self.cosine_similarity_matrix(mini_batch_list)
        self.plot_gr(occu_sim,"Heatmap of Occurence_similarity_matrix",'heatmap2.pdf') 


        

# Example usage:
def img_to_pdf(img_path,pdf_path):

    image = Image.open(img_path)
    pdf_bytes = img2pdf.convert(image.filename)
    file = open(pdf_path, "wb")
    file.write(pdf_bytes)
    image.close()
    file.close()

img_to_pdf("/home/hiddenmist/Aman_Lakshay/EDA/check_balancing.png","/home/hiddenmist/Aman_Lakshay/EDA/check_balancing.pdf")

# pdfs = ['output.pdf','class_EDA.pdf', 'check_balancing.pdf','quartile_analysis.pdf','mean_std_analysis.pdf','mean_quartile_analysis.pdf','heatmap1.pdf','heatmap2.pdf']
pdfs = ['output.pdf','class_EDA.pdf', 'check_balancing.pdf','quartile_analysis.pdf','mean_std_analysis.pdf','mean_quartile_analysis.pdf','heatmap1.pdf']


processor = EDA_items(path_to_csv)
processor.load_data()
processor.process_audio_data()
processor.plot_class_counts()
processor.quartile_class_count()
processor.mean_std_class()
processor.plot_mean_quartile()
processor.avg_adjacency_plot()
# processor.occurence_sim_plot()
# processor.plot_std_quartile()
# processor.print_cosine_similarity_matrix()
processor.make_pdf(pdfs)

