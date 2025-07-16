
def run_realtime_detection():
    import cv2
    import numpy as np
    from keras.models import load_model

    # === Load trained model ===
    model = load_model("models/mnist_cnn_model.h5")

    # === Open webcam ===
    cap = cv2.VideoCapture(0)
    print("🔍 Scanning... Press 'q' to quit.")

    # Initialize canvas with blank image
    default_canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas = default_canvas.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        canvas = default_canvas.copy()  # Reset canvas each frame

        # === Preprocessing ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # === Strict contour filtering ===
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            
            # Skip if:
            # - Too small or too large
            # - Aspect ratio not typical for digits (0.2-5.0)
            # - Area too small relative to bounding box
            if (w < 20 or h < 20 or w > 200 or h > 200 or 
                aspect_ratio < 0.2 or aspect_ratio > 5.0 or
                area < 0.2 * w * h):
                continue

            roi = thresh[y:y+h, x:x+w]
            
            # Additional check: digit should have reasonable white pixel density
            white_pixel_ratio = np.sum(roi == 255) / float(w * h)
            if white_pixel_ratio < 0.1 or white_pixel_ratio > 0.9:
                continue

            try:
                digit_resized = cv2.resize(roi, (18, 18))
            except:
                continue

            # === Digit processing ===
            canvas = np.zeros((28, 28), dtype=np.uint8)
            canvas[5:23, 5:23] = digit_resized

            # Center using image moments
            M = cv2.moments(canvas)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dx, dy = 14 - cx, 14 - cy
                transformation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                canvas = cv2.warpAffine(canvas, transformation_matrix, (28, 28))

            # Prepare input for model
            input_img = canvas.astype("float32") / 255.0
            input_img = input_img.reshape(1, 28, 28, 1)

            # Predict
            prediction = model.predict(input_img, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Only highlight if confidence is very high and it's a digit (0-9)
            if confidence > 0.95 and digit in range(10):
                # Create mask for the digit
                digit_mask = np.zeros_like(gray)
                cv2.drawContours(digit_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
                # Create green overlay only for the digit area
                green_overlay = np.zeros_like(frame)
                green_overlay[:] = (0, 255, 0)
                output = np.where(digit_mask[..., None] == 255, green_overlay, output)
                
                # Add prediction label
                cv2.putText(output, f"{digit} ({confidence*100:.0f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show windows - Model Input window is now larger (scaled up 10x)
        large_canvas = cv2.resize(canvas, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model Input (28x28) - Enlarged", large_canvas)
        cv2.imshow("Digit Highlight", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()