package com.nornenjs.android.dynamicview;

import android.app.Activity;
import android.media.Image;
import android.util.Log;
import android.view.*;
import android.view.ViewGroup.LayoutParams;
import android.view.inputmethod.EditorInfo;
import android.widget.*;
import android.widget.AbsListView.OnScrollListener;
import com.github.nkzawa.socketio.client.On;
import com.nineoldandroids.view.ViewPropertyAnimator;
import com.nornenjs.android.R;
import com.nornenjs.android.VolumeList;


public class PoppyViewHelper {

	private static final String TAG = "PoppyViewHelper";

	public enum PoppyViewPosition {
		TOP, BOTTOM
	};

	private static final int MAX_NUM_PER_PAGE = 10;

	private static final int SCROLL_TO_TOP = - 1;

	private static final int SCROLL_TO_BOTTOM = 1;

	private static final int SCROLL_DIRECTION_CHANGE_THRESHOLD = 5;

	private Activity mActivity;

	private LayoutInflater mLayoutInflater;

	private View mPoppyView;

	public LinearLayout title_bar;

	private int mScrollDirection = 0;

	private int mPoppyViewHeight = - 1;

	private PoppyViewPosition mPoppyViewPosition;

	VolumeList volumePage;

	//boolean bottom;

	public PoppyViewHelper(Activity activity, PoppyViewPosition position) {
		mActivity = activity;//mActivity 대체할 수도...
		volumePage = (VolumeList) activity;
		mLayoutInflater = LayoutInflater.from(activity);
		mPoppyViewPosition = position;
	}

	public PoppyViewHelper(Activity activity) {
		this(activity, PoppyViewPosition.TOP);
	}


	//for searchbar
	public View createPoppyViewOnListView(int titleview, int gridViewId, int poppyViewResId, OnScrollListener onScrollListener) {

		title_bar = (LinearLayout) mActivity.findViewById(titleview);

		final GridView gridView = (GridView)mActivity.findViewById(gridViewId);

//		if(gridView.getHeaderViewsCount() != 0) {
//			throw new IllegalArgumentException("use createPoppyViewOnListView with headerResId parameter");
//		}
//		if(gridView.getFooterViewsCount() != 0) {
//			throw new IllegalArgumentException("poppyview library doesn't support listview with toggle");
//		}

		Log.d(TAG, "createPoppyViewOnListView called");
		mPoppyView = mLayoutInflater.inflate(poppyViewResId, null);

		mPoppyView.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.d(TAG, "mPoppyView clicked");
				setImageSize(mPoppyView.getHeight());
				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(-mPoppyView.getHeight());
			}
		});

		initPoppyViewOnListView(title_bar, gridView, onScrollListener);

		return mPoppyView;
	}

	public void setImageSize(int ParentSize)
	{
		if(ParentSize != 0)
		{
			ImageView search_icon = (ImageView)mActivity.findViewById(R.id.search_image_icon);
			search_icon.getLayoutParams().width = ParentSize;
			search_icon.requestLayout();
		}
	}


	public View createPoppyViewOnListView(int editViewId,int gridViewId, int poppyViewResId) {
		return createPoppyViewOnListView(editViewId,gridViewId, poppyViewResId, null);
	}

	private void setPoppyViewOnView(View view) {

		LayoutParams lp = view.getLayoutParams();
		ViewParent parent = view.getParent();
		ViewGroup group = (ViewGroup)parent;
		int index = group.indexOfChild(view);

		final FrameLayout newContainer = new FrameLayout(mActivity);//FrameLayout

		group.removeView(view);
		group.addView(newContainer, index, lp);
		newContainer.addView(view);

		final FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT);
		layoutParams.gravity = mPoppyViewPosition == PoppyViewPosition.BOTTOM ? Gravity.BOTTOM : Gravity.TOP;
		newContainer.addView(mPoppyView, layoutParams);

		group.invalidate();
	}

	private void onScrollPositionChanged(int oldScrollPosition, int newScrollPosition) {
		int newScrollDirection;
		int ScrollDirection = newScrollPosition;
		System.out.println(oldScrollPosition + " ->" + newScrollPosition);

		if(newScrollPosition < oldScrollPosition) {
			newScrollDirection = SCROLL_TO_BOTTOM;
		} else {
			newScrollDirection = SCROLL_TO_TOP;
		}
		if(newScrollDirection != mScrollDirection) {
			mScrollDirection = newScrollDirection;
			translateYPoppyView(ScrollDirection);
		}
		//Log.d(TAG, "oldScrollPosition : " + oldScrollPosition + "newScrollPosition : " + newScrollPosition);
	}

	private void translateYPoppyView(int ScrollLocation) {
		setImageSize(mPoppyView.getHeight());

		mPoppyView.post(new Runnable() {

			//@Override
			public void run() {
				if (mPoppyViewHeight <= 0) {
					mPoppyViewHeight = mPoppyView.getHeight();
				}

				int translationY = 0;
				switch (mPoppyViewPosition) {
					case BOTTOM:
						translationY = mScrollDirection == SCROLL_TO_TOP ? 0 : mPoppyViewHeight;
						break;
					case TOP:
						translationY = mScrollDirection == SCROLL_TO_TOP ? -mPoppyViewHeight : 0;
						break;
				}

				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(translationY);//translationY
			}
		});
	}


	// for ListView

	int beforeCount = 0;
	private void initPoppyViewOnListView(LinearLayout title_bar, GridView gridView, final OnScrollListener onScrollListener) {
		setPoppyViewOnView(title_bar);

		gridView.setOnScrollListener(new OnScrollListener() {


			int mScrollPosition;

			//@Override
			public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount) {
				if(onScrollListener != null) {
					onScrollListener.onScroll(view, firstVisibleItem, visibleItemCount, totalItemCount);
				}
				View topChild = view.getChildAt(0);

				int newScrollPosition;
				if(topChild == null) {
					newScrollPosition = 0;
				} else {
					newScrollPosition = - topChild.getTop() + view.getFirstVisiblePosition() * topChild.getHeight();
				}

				if(newScrollPosition <= 160)
				{
					if(Math.abs(newScrollPosition - mScrollPosition) >= SCROLL_DIRECTION_CHANGE_THRESHOLD) {
						onScrollPositionChanged(mScrollPosition, newScrollPosition);
					}
				}

				mScrollPosition = newScrollPosition;

				Log.d(TAG, "totalItemCount : " + totalItemCount + ", beforeCount : " + beforeCount);
				if(totalItemCount >= MAX_NUM_PER_PAGE && (firstVisibleItem + visibleItemCount) ==  totalItemCount)
				{
					if(beforeCount != totalItemCount)//bottom이 제대로 될까..
					{
						Log.i("Info", "Scroll Bottom" );
						volumePage.getPage();
					}
					beforeCount = totalItemCount;
				}

			}

			//@Override
			public void onScrollStateChanged(AbsListView view, int scrollState) {
				if(onScrollListener != null) {
					onScrollListener.onScrollStateChanged(view, scrollState);
				}
			}

		});
	}
}
