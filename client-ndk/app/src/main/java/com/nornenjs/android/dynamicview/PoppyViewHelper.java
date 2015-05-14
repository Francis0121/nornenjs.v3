package com.nornenjs.android.dynamicview;

import android.app.Activity;
import android.util.Log;
import android.view.*;
import android.view.ViewGroup.LayoutParams;
import android.view.inputmethod.EditorInfo;
import android.widget.*;
import android.widget.AbsListView.OnScrollListener;
import com.nineoldandroids.view.ViewPropertyAnimator;
import com.nornenjs.android.VolumeList;


public class PoppyViewHelper {

	private static final String TAG = "PoppyViewHelper";

	public enum PoppyViewPosition {
		TOP, BOTTOM
	};

	private static final int SCROLL_TO_TOP = - 1;

	private static final int SCROLL_TO_BOTTOM = 1;

	private static final int SCROLL_DIRECTION_CHANGE_THRESHOLD = 5;

	private Activity mActivity;

	private LayoutInflater mLayoutInflater;

	private View mPoppyView;

	public EditText editView;

	private int mScrollDirection = 0;

	private int mPoppyViewHeight = - 1;

	private PoppyViewPosition mPoppyViewPosition;

	VolumeList volumePage;

	//public PoppyViewHelper(Activity activity, PoppyViewPosition position) {
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
		public View createPoppyViewOnListView(int editViewId, int listViewId, int poppyViewResId, OnScrollListener onScrollListener) {

		editView = (EditText) mActivity.findViewById(editViewId);
		editView.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(-mPoppyView.getHeight());
			}
		});

		editView.setOnEditorActionListener(new EditText.OnEditorActionListener(){
			@Override
			public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
				if (actionId == EditorInfo.IME_ACTION_SEARCH) {
					//performSearch();
					Log.d(TAG, "setOnEditorActionListener called");
					volumePage.searchRequest(editView.getText().toString());
					//return true;
				}
				return false;
			}
		});



		final ListView listView = (ListView)mActivity.findViewById(listViewId);

		if(listView.getHeaderViewsCount() != 0) {
			throw new IllegalArgumentException("use createPoppyViewOnListView with headerResId parameter");
		}
		if(listView.getFooterViewsCount() != 0) {
			throw new IllegalArgumentException("poppyview library doesn't support listview with footer");
		}

		mPoppyView = mLayoutInflater.inflate(poppyViewResId, null);
		initPoppyViewOnListView(editView, listView, onScrollListener);

		return mPoppyView;
	}


	public View createPoppyViewOnListView(int editViewId,int listViewId, int poppyViewResId) {
		return createPoppyViewOnListView(editViewId,listViewId, poppyViewResId, null);
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

	}

	private void translateYPoppyView(int ScrollLocation) {
		final int a = ScrollLocation;
		System.out.println("in translateYPoppyView : " + ScrollLocation);
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

				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(translationY);
			}
		});
	}

	private void translateYSearchView() {
		mPoppyView.post(new Runnable() {

			//@Override
			public void run() {
				if(mPoppyViewHeight <= 0) {
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

				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(translationY);
			}
		});
	}

	// for ListView

	private void initPoppyViewOnListView(EditText editView, ListView listView, final OnScrollListener onScrollListener) {
		setPoppyViewOnView(editView);
		listView.setOnScrollListener(new OnScrollListener() {

			//@Override
			public void onScrollStateChanged(AbsListView view, int scrollState) {
				if(onScrollListener != null) {
					onScrollListener.onScrollStateChanged(view, scrollState);
				}
			}

			int mScrollPosition;

			//@Override
			public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount) {
				if(onScrollListener != null) {
					onScrollListener.onScroll(view, firstVisibleItem, visibleItemCount, totalItemCount);
				}
				View topChild = view.getChildAt(0);

				int newScrollPosition = 0;
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


			}


		});
	}
}
