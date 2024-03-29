import streamlit as st
import cv2
from google.cloud import texttospeech_v1
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from io import BytesIO
import base64
import time
import numpy as np
import pygame

# Function for object detection and drawing bounding boxes
def predict_image_object_detection_sample(
    project: str,
    endpoint_id: str,
    image_content: bytes,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(image_content).decode("utf-8")
    instance = predict.instance.ImageObjectDetectionPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_object_detection_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=0.5,
        # Remove or adjust the max_predictions parameter
        max_predictions=None,  # Remove the restriction on the number of predictions
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions = response.predictions
    all_predictions = []
    for prediction in predictions:
        all_predictions.append(dict(prediction))

    return all_predictions

def draw_boxes(image, predictions, nms_threshold=0.5):
    final_detections = []
    for prediction in predictions:
        boxes = prediction['bboxes']
        labels = prediction['displayNames']
        confidences = prediction['confidences']
        detections = [(box, label, confidence) for box, label, confidence in zip(boxes, labels, confidences)]
        detections.sort(key=lambda x: x[2], reverse=True)
        while detections:
            box, label, confidence = detections.pop(0)
            final_detections.append((box, label, confidence))
            for i in range(len(detections)):
                box_i, _, _ = detections[i]
                iou = calculate_iou(box, box_i)
                if iou > nms_threshold:
                    detections.pop(i)
        for box, label, confidence in final_detections:
            height, width, _ = image.shape
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            color = (255, 0, 0)  # BGR color format (blue)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, final_detections

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def process_frames():
    st.title('Object Detection with Streamlit')
    st.markdown('Click the button below to start capturing frames from your webcam.')
    if st.button('Start Camera'):
        video_capture = cv2.VideoCapture(0)
        last_capture_time = time.time()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error('Error: Failed to capture frame.')
                break
            current_time = time.time()
            if current_time - last_capture_time >= 10:
                last_capture_time = current_time
                predictions = predict_image_object_detection_sample(
                    project="YOUR_PROJECT_ID",
                    endpoint_id="YOUR_ENDPOINT_ID",
                    image_content=cv2.imencode('.jpg', frame)[1].tobytes()
                )
                if predictions:
                    image_with_boxes, final_detections = draw_boxes(frame.copy(), predictions, nms_threshold=0.5)
                    st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)
                    client = texttospeech_v1.TextToSpeechClient()
                    for box, label, confidence in final_detections:
                        input_text = texttospeech_v1.SynthesisInput(text=f"{label}: {confidence:.2f}")
                        voice = texttospeech_v1.VoiceSelectionParams(
                            language_code="en-US",
                            name="en-US-Standard-C"
                        )
                        audio_config = texttospeech_v1.AudioConfig(
                            audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                            speaking_rate=1.0
                        )
                        response = client.synthesize_speech(
                            input=input_text, voice=voice, audio_config=audio_config
                        )
                        # Initialize a Sound object from the audio content using pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(BytesIO(response.audio_content))
                        # Play the audio
                        pygame.mixer.music.play()
            st.image(frame, caption='Camera Feed', use_column_width=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()

def main():
    process_frames()

if __name__ == '__main__':
    main()
