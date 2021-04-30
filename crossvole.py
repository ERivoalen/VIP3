import numpy as np
import time
import os
import streamlit as st
import cv2
import searchconsole
import tempfile
import networkx as nx
import matplotlib.pyplot as plt
    
def relations(lists):
    print(len(lists))
    result = []
    for component in range(0,len(lists)-1):
        for position in range(component+1, len(lists)):
            if(lists[component][3] < lists[position][1]):
                objects = [lists[component][4], " is above ", lists[position][4]]
        #print ('is above')
                result.append(objects)
            if(lists[component][1] > lists[position][3]):
                objects = [lists[component][4], " is under ",lists[position][4]]
        #print ('is under')
                result.append(objects)
            if(lists[component][0] > lists[position][2]):
                objects = [lists[component][4], " is right to ", lists[position][4]]
      #print ('is right to')
                result.append(objects)
            if(lists[component][2] < lists[position][0]):
                objects = [lists[component][4], " is left to ", lists[position][4]]
        #print ('is left to')
                result.append(objects)
            if(lists[component][0]>= lists[position][0] and lists[component][0]<lists[position][2]):
                objects = [lists[component][4], " is superimposing to ", lists[position][4]]
        #print ('is superposing to')
                result.append(objects)
            if(lists[component][2]> lists[position][0] and lists[component][0]<lists[position][0]):
                objects = [lists[component][4], " is superimposing to ", lists[position][4]]
        #print ('is superposing to')
                result.append(objects)
            if(lists[component][4][:len(lists[component][4])-1] == lists[position][4][:len(lists[position][4])-1]):
                if(lists[component][3] - lists[component][1]) > (lists[position][3] - lists[position][1]):
                    objects = [lists[component][4], " is taller than ", lists[position][4]]
          #print ('is talller than')
                    result.append(objects)
                if(lists[component][3] - lists[component][1]) < (lists[position][3] - lists[position][1]):
                    objects = [lists[component][4], " is smaller than ", lists[position][4]]
         #print ('is smaller than')
                    result.append(objects)
 
        return result

def main():

    title_info = st.empty()
    video_choose = st.empty()
    run_btn = st.empty()
    precision  = st.empty()
    image = st.empty() 
    confidence_label = st.empty()
    graph_display = st.empty()
    final_video = st.empty()

    title_info.title('Object Detection with YOLOv3')
    menu = ['Videos']

    choice = st.sidebar.selectbox('Dev Menu',menu)

    if choice == 'Videos':
        cfg_vid  = "/Users/erwanrivoalen/Downloads/darknet/build/darknet/x64/yolov3.cfg"
        image_vid  = video_choose.file_uploader('Select a video file', type=['mp4','mov'])
        
        names_vid  = "/Users/erwanrivoalen/Downloads/darknet/data/coco.names"
        weights_vid  = "/Users/erwanrivoalen/yolov3.weights"
        submit_vid = run_btn.button('Run')
        if submit_vid:
            print(image_vid.name)
            vn = image_vid.name
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(image_vid.read())
            vf = cv2.VideoCapture(tfile.name)
            #Ici il y avait un if
            video = vf
            writer = None
            h, w = None, None
            with open(names_vid) as f:
                labels = [line.strip() for line in f]
            network = cv2.dnn.readNetFromDarknet(cfg_vid,weights_vid)

            layers_names_all = network.getLayerNames()
            layers_names_output = \
            	[layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

            probability_minimum = 0.5

            threshold = 0.3
            colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
            lps = []
            f = 0
            t = 0
            while True:
            # Capturing frame-by-frame
               	ret, frame = video.read()

            # If the frame was not retrieved
            # e.g.: at the end of the video,
            # then we break the loop
                if not ret:
                    break

            # Getting spatial dimensions of the frame
            # we do it only once from the very beginning
            # all other frames have the same dimension
                if w is None or h is None:
                # Slicing from tuple only first two elements
                    h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         	swapRB=True, crop=False)


            # Implementing forward pass with our blob and only through output layers
            # Calculating at the same time, needed time for forward pass
                network.setInput(blob)  # setting blob as input to the network
                start = time.time()
                output_from_network = network.forward(layers_names_output)
                end = time.time()

            # Increasing counters for frames and total time
                f += 1
                t += end - start

            # Showing spent time for single current frame
                print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
                precision.write('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

                bounding_boxes = []
                confidences = []
                class_numbers = []
        
            # Going through all output layers after feed forward pass
                for result in output_from_network:
                
                # Going through all detections from current output layer
                    for detected_objects in result:
                        scores = detected_objects[5:]
                        class_current = np.argmax(scores)
                        confidence_current = scores[class_current]
                        if confidence_current > probability_minimum:
                            box_current = detected_objects[0:4] * np.array([w, h, w, h])

                        # Now, from YOLO data format, we can get top left corner coordinates
                        # that are x_min and y_min
                            x_center, y_center, box_width, box_height = box_current
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                            bounding_boxes.append([x_min, y_min,
                                               	int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)
                results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                       probability_minimum, threshold)
                yes = []
                confi = []
                rlist = []
                if len(results) > 0:
                # Going through indexes of results
                    for i in results.flatten():
                        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                        colour_box_current = colours[class_numbers[i]].tolist()
                    # Drawing bounding box on the original current frame
                        roi = frame[y_min:y_min+box_height+10,x_min:x_min+box_width+10]  
                        cv2.rectangle(frame, (x_min, y_min),
                                 	(x_min + box_width, y_min + box_height),
                                  	colour_box_current, 2)
                        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           	confidences[i])
                        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                	cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                        component = [x_min, y_min, x_min +box_width, y_min+box_height, labels[int(class_numbers[i])]]
                        yes.append(component)
                        confi.append(confidences[i])

                    for i in range(0,len(yes)-1):
                        yo = yes[i][4]
                        for j in range(i+1,len(yes)):
                            if yo == yes[j][4]:
                                yes[j][4] += f'{j}'

                    rlist = relations(yes)
                    if rlist:
                        for i in range (0,len(rlist)):
                            confidence_label.write('{0} with a confidence of : {1} {2} {3} with a confidence of : {4}'.format(rlist[i][0], confi[0], rlist[i][1], rlist[i][2], confi[1]))
                    
                        G = nx.DiGraph()
                        for triple in rlist:
                            G.add_node(triple[0])
                            G.add_node(triple[1])
                            G.add_node(triple[2])
                            G.add_edge(triple[0], triple[1])
                            G.add_edge(triple[1], triple[2])

                        pos = nx.spring_layout(G)
                        plt.figure()
                        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                           node_size=500, node_color='seagreen', alpha=0.9,
                           labels={node: node for node in G.nodes()})
                        plt.axis('off')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        graph_display.pyplot()   

                if writer is None:
            
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter('result'+vn, fourcc, 30,
                                         	    (frame.shape[1], frame.shape[0]), True)
            	#streamlit.image(frame)
                name = 'result'+vn
                image.image(frame)
                #writer.write(frame)
            print()
            print('Total number of frames', f)
            print('Total amount of time {:.5f} seconds'.format(t))
            print('FPS:', round((f / t), 1))

            video.release()
            writer.release()
           
                
            videofile = name
            print(videofile)
            m = os.path.splitext(videofile)[0]
            os.system('ffmpeg -i '+videofile+' -vcodec libx264 '+m+'fmpeg.mp4')
            final_video.video(m+'fmpeg.mp4')

if __name__ == '__main__':
    main()


