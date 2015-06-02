//package com.nornenjs.android.dynamicview;
//
//import android.app.Activity;
//import android.util.Log;
//import android.view.*;
//import android.view.ViewGroup.LayoutParams;
//import android.view.inputmethod.EditorInfo;
//import android.widget.*;
//import android.widget.AbsListView.OnScrollListener;
//import com.nineoldandroids.view.ViewPropertyAnimator;
//import com.nornenjs.android.VolumeList;
//
//
//public class SlidingViewHelper {
//
//    private static final String TAG = "SlidingViewHelper";
//
//    public enum SlidingViewPosition {
//        TOP, BOTTOM
//    };
//
//    private static final int SCROLL_TO_TOP = - 1;
//
//    private static final int SCROLL_TO_BOTTOM = 1;
//
//    private static final int SCROLL_DIRECTION_CHANGE_THRESHOLD = 5;
//
//    private Activity mActivity;
//
//    private LayoutInflater mLayoutInflater;
//
//    private View mSlidingView;
//
//    public EditText editView;
//
//    private int mScrollDirection = 0;
//
//    private int mSlidingViewHeight = - 1;
//
//    private SlidingViewPosition mSlidingViewPosition;
//
//    //VolumeList volumePage;
//
//    //boolean bottom;
//
//    public SlidingViewHelper(Activity activity, SlidingViewPosition position) {
//        mActivity = activity;//mActivity 대체할 수도...
//        //volumePage = (VolumeList) activity;
//        mLayoutInflater = LayoutInflater.from(activity);
//        mSlidingViewPosition = position;
//    }
//
//    public SlidingViewHelper(Activity activity) {
//        this(activity, SlidingViewPosition.BOTTOM);
//    }
//
//
//    //for searchbar
//    public View createSlidingViewOnVolumeView(int volumeLayout, int slidingViewResId, OnScrollListener onScrollListener) {
//
//        //sliding layout id가 poppyViewResId 위치
//        //여기서 슬라이딩 창의 이벤트 리스너
//
////        editView = (EditText) mActivity.findViewById(editViewId);//editview가 아니라 poppyViewResId를 터치했을때...
////        editView.setOnClickListener(new View.OnClickListener() {
////            @Override
////            public void onClick(View v) {
////                Log.d(TAG, "setOnClickListener called : " + v.getId() + ", editViewId : " + editView.getId() + ", mPoppyView : " + mSlidingView.getId());
////                ViewPropertyAnimator.animate(mSlidingView).setDuration(300).translationY(-mSlidingView.getHeight());
////            }
////        });
//
//        mSlidingView = mLayoutInflater.inflate(slidingViewResId, null);
////        initSlidingViewOnListView(editView, listView, onScrollListener);
//
//        return mSlidingView;
//    }
//
//
//    public View createSlidingViewOnVolumeView(int volumeLayout, int poppyViewResId) {
//        return createSlidingViewOnVolumeView(volumeLayout, poppyViewResId, null);
//    }
//
//    private void setSlidingViewOnView(View view) {
//        LayoutParams lp = view.getLayoutParams();
//        ViewParent parent = view.getParent();
//        ViewGroup group = (ViewGroup)parent;
//        int index = group.indexOfChild(view);
//
//        final FrameLayout newContainer = new FrameLayout(mActivity);//FrameLayout
//
//
//        group.removeView(view);
//
//        group.addView(newContainer, index, lp);
//
//        newContainer.addView(view);
//
//        Log.d(TAG, "addView called");
//
//        final FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT);
//        layoutParams.gravity = mSlidingViewPosition == SlidingViewPosition.BOTTOM ? Gravity.BOTTOM : Gravity.TOP;
//        newContainer.addView(mSlidingView, layoutParams);
//
//        group.invalidate();
//    }
//
//
//    private void translateYSlidingView(int ScrollLocation) {
//        final int a = ScrollLocation;
//        System.out.println("in translateYPoppyView : " + ScrollLocation);
//        mSlidingView.post(new Runnable() {
//
//            //@Override
//            public void run() {
//                if (mSlidingViewHeight <= 0) {
//                    mSlidingViewHeight = mSlidingView.getHeight();
//                }
//
//                int translationY = 0;
//                switch (mSlidingViewPosition) {
//                    case BOTTOM:
//                        translationY = mScrollDirection == SCROLL_TO_TOP ? 0 : mSlidingViewHeight;//
//                        break;
//                    case TOP:
//                        translationY = mScrollDirection == SCROLL_TO_TOP ? -mSlidingViewHeight : 0;
//                        break;
//                }
//
//                Log.d(TAG, "animate view!");
//                ViewPropertyAnimator.animate(mSlidingView).setDuration(300).translationY(translationY);//translationY
//            }
//        });
//    }
//
//
//    // for ListView
//
//    private void initSlidingyViewOnVolumeView(EditText editView, ListView listView, final OnScrollListener onScrollListener) {
//        setSlidingViewOnView(editView);
//        Log.d(TAG, "editView.getheight : " + editView.getHeight());
//
//    }
//}
