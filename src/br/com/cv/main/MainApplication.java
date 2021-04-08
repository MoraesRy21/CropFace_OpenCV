package br.com.cv.main;

import org.opencv.core.Core;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;
import javafx.scene.image.Image;

/**
 * This application handles a video stream and try to find any possible human
 * face in a frame and crop the face to save in a specific folder. 
 * It can use the Haar or the LBP classifier.
 *
 * @author Iarley Moraes
 * @version 2.0 (2021-04-04)
 * @since 1.0 (2020-01-20)
 *
 */
public class MainApplication extends Application {

    @Override
    public void start(Stage primaryStage) {
        try {
            // load the FXML resource
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/br/com/cv/view/FaceDetection.fxml"));
            BorderPane root = (BorderPane) loader.load();
            
            root.setStyle("-fx-background-color: whitesmoke;");
            Scene scene = new Scene(root, 800, 600);
            scene.getStylesheets().add(getClass().getResource("/br/com/cv/view/application.css").toExternalForm());
            
            Image icon = new Image("file:resources\\icon\\faceDetectionIcon.png");
            
            primaryStage.getIcons().add(icon);
            primaryStage.setTitle("Crop Face Image");
            primaryStage.setScene(scene);
            primaryStage.show();
            
            // init the controller
            FaceDetectionController controller = loader.getController();
            controller.init();

            // set the proper behavior on closing the application
            primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>() {
                @Override
                public void handle(WindowEvent we) {
                    controller.setClosed();
                }
            }));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // load the native OpenCV library
        System.load(System.getProperty("user.dir")+"\\lib\\x64\\"+Core.NATIVE_LIBRARY_NAME+".dll");
        launch(args);
    }
}
