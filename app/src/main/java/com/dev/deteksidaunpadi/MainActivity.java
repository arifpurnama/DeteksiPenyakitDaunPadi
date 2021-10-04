package com.dev.deteksidaunpadi;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.theartofdev.edmodo.cropper.CropImage;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    //TensorFlowLite
    protected Interpreter tflite;
    private TensorImage inputImageBuffer;
    private int imageSizeX;
    private int imageSizeY;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private Bitmap bitmap;
    private List<String> labels;

    ImageView imageView;
    Button btnDeteksi, btnCariGambar, btnReset;
    TextView classitext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.ivNoImage);
        btnDeteksi = findViewById(R.id.btnDeteksi);
        btnCariGambar = findViewById(R.id.btnCariGambar);
        btnReset = findViewById(R.id.btnReset);
        classitext = findViewById(R.id.txtResult);

        //Aksi Btn Deteksi
        btnDeteksi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Reads type and shape of input and output tensors, respectively. [START]
                int imageTensorIndex = 0;
                int probabilityTensorIndex = 0;

                int[] inputImageShape = tflite.getInputTensor(imageTensorIndex).shape();
                DataType inputDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                int[] outputImageShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
                DataType outputDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

                imageSizeY = inputImageShape[1];
                imageSizeX = inputImageShape[2];
                // Reads type and shape of input and output tensors, respectively. [END]

                // Creates the input tensor.
                inputImageBuffer = new TensorImage(inputDataType);
                // Membuat kontainer untuk hasil
                outputProbabilityBuffer = TensorBuffer.createFixedSize(outputImageShape, outputDataType);
                // Creates the post processor for the output probability.
                probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

                //load gambar bitmap
                inputImageBuffer = loadImage(bitmap);
                //menjalankan model
                tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                //menampilkan hasil
                showresult();
            }
        });

        //btnAksi SearcImage
        btnCariGambar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Activity CropImage
                CropImage.activity().start(MainActivity.this);
            }
        });

        btnReset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Activity Reset
                imageView.setImageResource(R.drawable.no_image_available);
                classitext.setText("Hasil");
                imageView.destroyDrawingCache();
                btnDeteksi.setEnabled(false);
            }
        });

        try {
            tflite = new Interpreter(loadmodelfile(this));
        } catch (Exception e) {
            Log.e("tfliteSupport", "Error reading model file", e);
        }
    }

    //Load Image
    private TensorImage loadImage(final Bitmap bitmap) {
        inputImageBuffer.load(bitmap);
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    //load model tflite
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("penyakitdaunpadi.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    //Menampilkan Hasil
    private void showresult() {
        try {
            //Load Label
            labels = FileUtil.loadLabels(this, "penyakit_label.txt");
        } catch (Exception e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            if (entry.getValue() == maxValueInMap) {
                classitext.setText(entry.getKey());
            }
        }
    }

    //Activity Crop Image
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                if (result != null) {
                    Uri uri = result.getUri();
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                        imageView.setImageBitmap(bitmap);
                        btnDeteksi.setEnabled(true);
                    } catch (IOException e) {
                        Log.e("ImageSupport", "Error reading Image file", e);
                    }
                } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                    Exception error = result.getError();
                    Log.e("CropError", error + "");
                }
            }
        }
    }

}