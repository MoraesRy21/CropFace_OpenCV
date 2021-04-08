package br.com.cv.main;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import br.com.cv.utils.Utils;
import java.io.File;
import javafx.application.Platform;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.control.TextField;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 *
 */
public class FaceDetectionController {
    // FXML buttons

    @FXML
    private Button cameraButton;
    // the FXML area for showing the current frame
    @FXML
    private ImageView originalFrame;
    // checkboxes for enabling/disabling a classifier
    @FXML
    private CheckBox haarClassifier;
    @FXML
    private CheckBox lbpClassifier;
    @FXML
    private TextField pathField;
    @FXML
    private TextField initialDelay;
    @FXML
    private TextField cropPeriod;
    @FXML
    private TextField photoCount;
    @FXML
    private Label labelCounter;
    @FXML
    private TextField fileName;

    private ScheduledExecutorService timer; // a timer for acquiring the video stream
    private ScheduledExecutorService timerToCrop;
    private VideoCapture capture; // the OpenCV object that performs the video capture
    private boolean cameraActive; // a flag to change the button behavior

    private CascadeClassifier faceCascade; // face cascade classifier
    private int absoluteFaceSize;
    
    private int countPhoto;
    private int delay;
    private int period;
    private String path = System.getProperty("user.dir");
    private String insertPath;
    private String insertFileName;
    
    boolean possoVerificar = false;
    
    public synchronized void setPossoVerificar(boolean possoVerificar){
        this.possoVerificar = possoVerificar;
    }
    
    /**
     * Init the controller, at start time
     */
    protected void init() {
        this.capture = new VideoCapture();
        this.faceCascade = new CascadeClassifier();
        this.absoluteFaceSize = 0;

        // set a fixed width for the frame
        originalFrame.setFitWidth(600);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);
    }

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera() {
        if (!this.cameraActive && checkFilds()) {
            // disable setting checkboxes
            this.haarClassifier.setDisable(true);
            this.lbpClassifier.setDisable(true);
            this.pathField.setDisable(true);
            
            this.initialDelay.setDisable(true);
            this.cropPeriod.setDisable(true);
            this.photoCount.setDisable(true);

            // start the video capture
            this.capture.open(0);
            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run() {
                        // effectively grab and process a single frame
                        Mat frame = grabFrame();
                        // convert and show the frame
                        Image imageToShow = Utils.mat2Image(frame);
                        updateImageView(originalFrame, imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, (long) 33, TimeUnit.MILLISECONDS);
                
                this.timerToCrop = Executors.newSingleThreadScheduledExecutor();
                this.timerToCrop.scheduleWithFixedDelay(new ScheduleTask(), delay, period, TimeUnit.SECONDS);
                

                // update the button content
                this.cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable classifiers checkboxes
            this.haarClassifier.setDisable(false);
            this.lbpClassifier.setDisable(false);
            this.pathField.setDisable(false);
            
            this.initialDelay.setDisable(false);
            this.cropPeriod.setDisable(false);
            this.photoCount.setDisable(false);
            
            photoCount.setText(Integer.toString(countPhoto));

            // stop the timer
            this.stopAcquisition();
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Mat grabFrame() {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    Core.flip(frame, frame, 1);
                    // face detection
                    this.detectAndDisplay(frame);
                }else
                    System.out.println("frame null");

            } catch (Exception e) {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return frame;
    }
    
    /**
     * Method for face detection and tracking
     *
     * @param frame it looks for faces in this frame
     */
    private void detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();
        
        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
       
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);
     
        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0) {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0) {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }
        
        // detect faces
        this.faceCascade.detectMultiScale(grayFrame, faces, 1.05, 7, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size(224, 224));

        cropAndSave(faces, frame);
        
        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.toArray();
        if(facesArray.length > 1)
            System.out.println("facesArray = "+facesArray.length);
        
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
        }
        
    }
    
    public void cropAndSave(MatOfRect faces, Mat frame){
        if(possoVerificar && !faces.empty()){
            System.out.println("faces not empty!");
            for (Rect faceRectangle : faces.toArray()) {
                Mat faceImage = frame.submat(faceRectangle);
                String fileName = insertPath+"\\"+ insertFileName+ "_" + countPhoto + ".jpg";
                Imgcodecs.imwrite(fileName, faceImage);
                System.out.println("Tirei foto: Face_" + countPhoto);
                countPhoto++;
                Platform.runLater(alterLabebl);
            }
            setPossoVerificar(false);
        }
    }
    
    Runnable alterLabebl = new Runnable(){
        @Override
        public void run() {
            labelCounter.setText(""+countPhoto);
        }
    };
    
    class ScheduleTask implements Runnable {
        @Override
        public void run() {
            setPossoVerificar(true);
            try {
                while(possoVerificar){
                    Thread.sleep(100);
                }
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
        }
    }
    
    public boolean checkFilds(){
        String path = pathField.getText();
        if(path != null && !path.equals("")){
            File file = new File(path);
            if(!file.isDirectory()){
                JOptionPane.showMessageDialog(null, "Path incorrect!");
                return false;
            }
        }
        insertPath = path;
        System.out.println(insertPath);
        
        String fileName = this.fileName.getText();
        if(fileName == null || fileName.equals("")){
            JOptionPane.showMessageDialog(null, "File name empty!");
            return false;
        }
        insertFileName = fileName;
        
        String initD = initialDelay.getText(), cropP = cropPeriod.getText(), photoC = photoCount.getText();
        if((initD != null && !initD.equals("")) || (cropP != null && !cropP.equals("")) || (photoC != null && !photoC.equals(""))){
            try{
                countPhoto = Integer.parseInt(photoC);
                delay = Integer.parseInt(initD);
                period = Integer.parseInt(cropP);
            }catch(Exception ex){
                JOptionPane.showMessageDialog(null, "Values in the left side fields incorrects!");
                return false;
            }
        }else{
            JOptionPane.showMessageDialog(null, "Values in the left side fields incorrects!");
            return false;
        }
        System.out.println(countPhoto);
        System.out.println(delay);
        System.out.println(period);
        
        return true;
    }

    /**
     * The action triggered by selecting the Haar Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void haarSelected(Event event) {
        // check whether the lpb checkbox is selected and deselect it
        if (this.lbpClassifier.isSelected()) {
            this.lbpClassifier.setSelected(false);
        }
        this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
    }

    /**
     * The action triggered by selecting the LBP Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void lbpSelected(Event event) {
        // check whether the haar checkbox is selected and deselect it
        if (this.haarClassifier.isSelected()){
            this.haarClassifier.setSelected(false);
        }
        this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
    }

    /**
     * Method for loading a classifier trained set from disk
     *
     * @param classifierPath the path on disk where a classifier trained set is
     * located
     */
    private void checkboxSelection(String classifierPath) {
        // load the classifier(s)
        this.faceCascade.load(classifierPath);

        if (!this.haarClassifier.isSelected() && !this.lbpClassifier.isSelected()) {
            this.cameraButton.setDisable(true);
        } else { // now the video capture can start
            this.cameraButton.setDisable(false);
        }
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
                setPossoVerificar(false);
                this.timerToCrop.shutdown();
                this.timerToCrop.awaitTermination(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view the {@link ImageView} to update
     * @param image the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

}
