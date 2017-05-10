package com.example.zte.testfacedemo;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Message;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.ZTEFace;
import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.appindexing.Thing;
import com.google.android.gms.common.api.GoogleApiClient;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private JavaCameraView cameraView;
    private Mat mRgba;
    //相对的脸的大小。
    private float mRelativeFaceSize = 0.2f;
    //绝对的脸的大小。
    private int mAbsoluteFaceSize = 0;
    private int num = 0;
    private boolean isFinished = false;
    /**持续获取到人脸数量后，就使用当前截图**/
    public static final int FACE_SUCCESS_COUNT = 10;
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    /**项目目录*/
    public static final String PROJECT_FOLDER_PATH = getSDCardRootPath() + File.separator + "Jingwutong" + File.separator;
    /**缓存路径，用于存放临时的图片*/
    public static final String CASH = PROJECT_FOLDER_PATH + "cashes" + File.separator;
    /**缓存用的单张图片**/
    public static final String CASH_IMG = CASH+"cash.jpg";
    private static final String TAG = "111";
    private CascadeClassifier mJavaDetector;
    TextView tv1;
    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    private GoogleApiClient client;
    String sdPath= Environment.getExternalStorageDirectory().getAbsolutePath()+"/Jingwutong";

    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        /**
         * 只有当OpenCVManager连接成功，我们才能使用openCV的一些功能。
         * @param status
         */
        @Override
        public void onManagerConnected(int status){
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.e(TAG, "OpenCV loaded successfully" );
                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier" );
                            mJavaDetector = null;
                        } else {
                            Log.e(TAG, "Loaded cascade classifier from"+ mCascadeFile.getAbsolutePath() );
                        }
                        cascadeDir.delete();
                        //getLocalHeadMat();
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown:" );
                    }
                    //当放回值是成功的时候，就显示
                    cameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
    private byte[] mFea1;
    private byte[] mFea2;
    private Bitmap bmp;
    private ThreadPoolExecutor poolExecutor;

    private void initViews() {
        cameraView = (JavaCameraView) findViewById(R.id.main_camera);
        cameraView.setCvCameraViewListener(this);
        //使用前置摄像头来获取图像。
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
    }
    private static final int REQUEST_CODE = 0; // 请求码
    // 所需的全部权限
    static final String[] PERMISSIONS = new String[]{
            android.Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.CAMERA,
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initData();
    }


    private void initData(){
        initViews();
        poolExecutor = new ThreadPoolExecutor(3, 5,
                1, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>(128));
        String strPicName = sdPath+"/pic.rgb8";
        byte[] b3 = null;
        try {

            InputStream inputStream =  getResources().openRawResource(R.raw.pic);
            b3 = new byte[inputStream.available()];
            Log.e(TAG, "onCreate: "+inputStream.available() );
            inputStream.read(b3,0,inputStream.available());
            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        //加载模型
        ZTEFace.loadModel(sdPath+"/d.dat",sdPath+"/a.dat",sdPath+"/db.dat",sdPath+"/p.dat");
        //人脸检测 输入 rgb8或YUV
        byte[] ret = ZTEFace.detectFace(b3,100,123);
//        //人脸信息转换
        ZTEFace.FaceInfo aa = ZTEFace.getFaceInfo(ret);
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    private byte[] getBytes(byte[] bb, ZTEFace.FacePointInfo facePointInfo,int width,int height) {
        int eyeLeftX = (int) facePointInfo.ptEyeLeft.x;
        int eyeLeftY= (int) facePointInfo.ptEyeLeft.y;
        int eyeRightX= (int) facePointInfo.ptEyeRight.x;
        int eyeRightY= (int) facePointInfo.ptEyeRight.y;
        int eyeNoseX= (int) facePointInfo.ptNose.x;
        int eyeNoseY =(int) facePointInfo.ptNose.y;
        int mouthLeftX=(int) facePointInfo.ptMouthLeft.x;
        int mouthLeftY = (int) facePointInfo.ptMouthLeft.y;
        int mouthRightX = (int) facePointInfo.ptMouthRight.x;
        int mouthRightY = (int) facePointInfo.ptMouthRight.y;
        //特征提取
        return ZTEFace.getFea(bb,width,height,eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, eyeNoseX, eyeNoseY, mouthLeftX, mouthLeftY, mouthRightX, mouthRightY);
    }

    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    public Action getIndexApiAction() {
        Thing object = new Thing.Builder()
                .setName("Main Page") // TODO: Define a title for the content shown.
                // TODO: Make sure this auto-generated URL is correct.
                .setUrl(Uri.parse("http://[ENTER-YOUR-URL-HERE]"))
                .build();
        return new Action.Builder(Action.TYPE_VIEW)
                .setObject(object)
                .setActionStatus(Action.STATUS_TYPE_COMPLETED)
                .build();
    }

    @Override
    public void onStart() {
        super.onStart();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client.connect();
        AppIndex.AppIndexApi.start(client, getIndexApiAction());
    }
    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG,"OpenCV library not found!");
            //没有库文件久启动下载
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.e(TAG, "OpenCV library found inside package. Using it!");
            //有库文件久使用它。
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    @Override
    public void onStop() {
        super.onStop();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        AppIndex.AppIndexApi.end(client, getIndexApiAction());
        client.disconnect();
    }

    /**
     * 获取相同尺寸的头像
     *
     * @param mat 传入的mat
     * @return Mat
     */
    /**上传到后台对比图片的尺寸**/
    public static final int COMPARE_SIZE = 320;
    private Mat getDefaultCompareSize(Mat mat) {
        Mat result = new Mat();
        Size size = new Size(COMPARE_SIZE, COMPARE_SIZE);
        Imgproc.resize(mat, result, size);
        return result;
    }

    public static String getSDCardRootPath() {
        if (avaiableSDCard()) {
            return getSDCardRoot().getPath();
        }
        return null;
    }
    public static boolean avaiableSDCard() {
        String status = Environment.getExternalStorageState();
        if (status.equals(Environment.MEDIA_MOUNTED)) {
            return true;
        } else {
            return false;
        }
    }
    public static File getSDCardRoot() {
        if (avaiableSDCard()) {
            return Environment.getExternalStorageDirectory();
        }
        return null;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }
    private byte[] bitmap2Byte(Bitmap bitmap) {
        //将图片资源转换成字节流
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        byte[] buffer = baos.toByteArray();
        try {
            baos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return buffer;
    }

    private byte[] getFea(byte[] buffer) {
        byte[] ret = ZTEFace.detectFace(buffer,100,123);
        Log.e(TAG, "getFea: "+ret.length );
        ZTEFace.FaceInfo aa = ZTEFace.getFaceInfo(ret);

        int eyeLeftX = (int)aa.info[0].ptEyeLeft.x;
        int eyeLeftY= (int)aa.info[0].ptEyeLeft.y;
        int eyeRightX= (int)aa.info[0].ptEyeRight.x;
        int eyeRightY= (int)aa.info[0].ptEyeRight.y;
        int eyeNoseX= (int)aa.info[0].ptNose.x;
        int eyeNoseY =(int)aa.info[0].ptNose.y;
        int mouthLeftX=(int)aa.info[0].ptMouthLeft.x;
        int mouthLeftY = (int)aa.info[0].ptMouthLeft.y;
        int mouthRightX = (int)aa.info[0].ptMouthRight.x;
        int mouthRightY = (int)aa.info[0].ptMouthRight.y;
        //特征提取
        byte[] fea =  ZTEFace.getFea(buffer,100,123,eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, eyeNoseX, eyeNoseY, mouthLeftX, mouthLeftY, mouthRightX, mouthRightY);
//特征比对
        return fea;
    }
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        //使用前置摄像头时旋转90度
        Mat rotateMat = Imgproc.getRotationMatrix2D(new Point(mRgba.cols() / 2, mRgba.rows() / 2), 90, 1);
        Imgproc.warpAffine(mRgba, mRgba, rotateMat, mRgba.size());

        if (mAbsoluteFaceSize == 0) {
            int height = mRgba.rows();
            Log.e(TAG, "onCameraFrame: "+height );
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();
        if (mJavaDetector != null) {
            mJavaDetector.detectMultiScale(mRgba, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        //检测并且显示。
        Rect[] facesArray = faces.toArray();
        if (facesArray.length > 0 && !isFinished ) {
            num++;
            if (num % FACE_SUCCESS_COUNT == 0) {
                Log.e(TAG, "onCameraFrame: "+"截取人脸"+"num的值"+num );
                Mat mat = getDefaultCompareSize(mRgba.submat(facesArray[0]));
                bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);

                if(bmp != null){
                    Runnable runnable =  new Runnable() {
                        @Override
                        public void run() {
                            long start = System.currentTimeMillis();
                            byte[] bytes = bitmap2Byte(bmp);
                            mFea1 = getFea(bytes);
                            long end = System.currentTimeMillis();
                            Log.e(TAG, "本次获取所消耗的时间: "+(end - start) );
                        }
                    };
                    poolExecutor.execute(runnable);
                }

//                compare(mat,headMat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY, 3);
                Imgcodecs.imwrite(CASH_IMG, mat);
                Bitmap bitmap = BitmapFactory.decodeFile(CASH_IMG);
                byte[] bytes = null;
            }
            //在这里画矩形
            Imgproc.rectangle(mRgba, facesArray[0].tl(), facesArray[0].br(), FACE_RECT_COLOR, 3);
            Log.e(TAG, "画边框 "+facesArray.length );
        } else {
            //当外框截取失败（即facesArray.length == 0）的时候就将num归零。
            num = 0;
        }

        return mRgba;
    }

    public void compareClick(View view) {
        isFinished = true;
        byte[] fea3 = null;
        if(bmp != null){
            byte[] bytes = bitmap2Byte(bmp);
            fea3 = getFea(bytes);
            Log.e(TAG, "compareClick: "+fea3.length );
        }

        FileInputStream ins;
        byte[] b1 = null;
        try {

            File file = new File(CASH + "cash.jpg");
            Log.e(TAG, "onCameraFrame: " );
            b1= new byte[(int) file.length()];
            Log.e(TAG, "onCreate: "+file.length());
            Log.e(TAG, "onCreate: "+file.getAbsolutePath() );
            ins= new FileInputStream(file);
            ins.read(b1,0, (int) file.length());
            ins.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        byte[] ret1 = ZTEFace.detectFace(b1,160,160);
        ZTEFace.FaceInfo a1 = ZTEFace.getFaceInfo(ret1);
        mFea1 = getBytes(b1, a1.info[0],160,160);
        Log.e(TAG, "compareClick: "+mFea1.length  );
//        Log.e(TAG, "compareClick: "+mFea2.length );
        float fScore = ZTEFace.feaCompare(fea3,mFea1);
        tv1 = (TextView) findViewById(R.id.tv1);
        tv1.setText("比对所得分数"+fScore);
    }

    public void intent2Scan(View view) {
        startActivity(new Intent(this,SuspectScanActicity.class));
    }
}
