# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:57:02 2021
@author: andy
"""



from  matplotlib  import  animation
import matplotlib.pyplot as plt
import os

class Gif():
    def __init__(self, path = './', file_name = ''):
        self.frames = []
        self.file_name = path + file_name
        self.iter = 1
        if not os.path.exists(path):
            os.makedirs(path)
    
    def load_frame(self, frame):
        self.frames.append(frame)
        
    def save_gif(self):
        #Mess with this to change frame size
        plt.figure(figsize=(self.frames[0].shape[1] / 72.0, self.frames[0].shape[0] / 72.0), dpi=72)
    
        patch = plt.imshow(self.frames[0])
        plt.axis ('OFF')
    
        def animate(i):
            patch.set_data(self.frames[i])
    
        anim = animation.FuncAnimation(plt.gcf(), animate , frames = len(self.frames), interval = 1)
        anim.save(self.file_name + '{}.gif'.format(self.iter), writer='pillow', fps=60)
        self.iter += 1
        self.frames = []
        plt.close()