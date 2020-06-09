package com.example.android.imagegallery;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Size;
import android.util.TypedValue;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.android.imagegallery.env.BorderedText;
import com.example.android.imagegallery.tflite.Classifier;
import com.example.android.imagegallery.tracking.MultiBoxTracker;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    final int GALLERY_REQUEST_CODE = 1;
    public Classifier detector;
    final DetectorMode MODE = DetectorMode.TF_OD_API;
    ImageView imageView;
    org.tensorflow.lite.examples.detection.customview.OverlayView trackingOverlay;
    Button galleryButton, cropButton, inferenceButton;
    Bitmap bitmap, bitmapcrop, bitmapinference;
    Context myContext = getApplication();
    // Configuration values for the prepackaged SSD model.
    final int TF_OD_API_INPUT_SIZE = 300;
    final boolean TF_OD_API_IS_QUANTIZED = false;
    final String TF_OD_API_MODEL_FILE = "onioncam.tflite";
    final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelsonion.txt";
    final float MINIMUM_CONFIDENCE_TF_OD_API = 0.8f;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView=(ImageView)findViewById(R.id.imgview);
        galleryButton = (Button)findViewById(R.id.buttonGallery);
        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                pickFromGallery();
            }
        });
        cropButton = (Button)findViewById(R.id.buttonCrop);
        cropButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                cropImage();
            }
        });
        inferenceButton = (Button)findViewById(R.id.buttonInference);
        inferenceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                runinference();
            }
        });


        trackingOverlay = (org.tensorflow.lite.examples.detection.customview.OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                    }
                });
        tracker = new MultiBoxTracker(this);
        tracker.setFrameConfiguration(300, 300, 0);


        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 20, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);


        try{
            detector = com.example.android.imagegallery.tflite.TFLiteObjectDetectionAPIModel.create(
                    getAssets(),
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED);}
        catch (final IOException e) {
            e.printStackTrace();
            Toast.makeText(
                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT).show();
        }

    }



    public void runinference(){
        // Minimum detection confidence to track a detection.
        //instantiate a tracking view to draw our results
        long timestamp = 0;
        long currTimestamp = timestamp;
        bitmapinference = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888);
        //do inference
        final List<Classifier.Recognition> results = detector.recognizeImage(bitmapcrop);
        try{
                final Canvas canvas = new Canvas(bitmapcrop);
                final Paint paint = new Paint();
                paint.setColor(Color.GREEN);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(3.0f);
                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();

                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= minimumConfidence) {

                        canvas.drawRect(location, paint);
                        result.setLocation(location);
                        mappedRecognitions.add(result);

                    }}
                tracker.trackResults(mappedRecognitions, currTimestamp);
                tracker.draw(canvas);
                //trackingView.postInvalidate();
                //canvas.drawBitmap();
            }
        catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(
                    getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT).show();
        }
    }

    private enum DetectorMode {
        TF_OD_API;
    }
    /*@Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }*/
    private void pickFromGallery(){
        //Create an Intent with action as ACTION_PICK
        Intent intent=new Intent(Intent.ACTION_PICK);
        // Sets the type as image/*. This ensures only components of type image are selected
        intent.setType("image/*");
        //We pass an extra array with the accepted mime types. This will ensure only components with these MIME types as targeted.
        String[] mimeTypes = {"image/jpeg", "image/png"};
        intent.putExtra(Intent.EXTRA_MIME_TYPES,mimeTypes);
        // Launching the Intent
        startActivityForResult(intent,GALLERY_REQUEST_CODE);
    }
    private void cropImage(){
        bitmapcrop=Bitmap.createBitmap(bitmap, 0,0,300, 300);
        imageView.setImageBitmap(bitmapcrop);

    };
    public void onActivityResult(int requestCode,int resultCode,Intent data){
        // Result code is RESULT_OK only if the user selects an Image
        if (resultCode == Activity.RESULT_OK)
            switch (requestCode){
                case GALLERY_REQUEST_CODE:
                    //data.getData return the content URI for the selected Image
                    Uri selectedImage = data.getData();
                    String[] filePathColumn = { MediaStore.Images.Media.DATA };
                    // Get the cursor
                    Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                    // Move to first row
                    cursor.moveToFirst();
                    //Get the column index of MediaStore.Images.Media.DATA
                    int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                    //Gets the String value in the column
                    String imgDecodableString = cursor.getString(columnIndex);
                    cursor.close();
                    // Set the Image in ImageView after decoding the String
                    bitmap = BitmapFactory.decodeFile(imgDecodableString);
                    imageView.setImageBitmap(bitmap);

                    break;

            }
    }
}