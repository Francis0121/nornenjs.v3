package com.nornenjs.android.dynamicview;

import android.app.Activity;
import android.util.Log;
import android.view.*;
import android.view.ViewGroup.LayoutParams;
import android.view.inputmethod.EditorInfo;
import android.widget.*;
import android.widget.AbsListView.OnScrollListener;
import com.github.nkzawa.socketio.client.On;
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

	//boolean bottom;

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

		editView = (EditText) mActivity.findViewById(editViewId);//editview가 아니라 poppyViewResId를 터치했을때...
//		editView.setOnClickListener(new View.OnClickListener() {
//			@Override
//			public void onClick(View v) {
//				Log.d(TAG, "setOnClickListener called : " + v.getId() + ", editViewId : " + editView.getId() + ", mPoppyView : " + mPoppyView.getId());
//				//ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(-mPoppyView.getHeight());
//				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(mPoppyView.getHeight());
//			}
//		});

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
			throw new IllegalArgumentException("poppyview library doesn't support listview with toggle");
		}

		Log.d(TAG, "createPoppyViewOnListView called");
		mPoppyView = mLayoutInflater.inflate(poppyViewResId, null);

			mPoppyView.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					Log.d(TAG, "mPoppyView clicked");
					ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(-mPoppyView.getHeight());
				}
			});

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

		Log.d(TAG, "addView called");

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
			Log.d(TAG, "onScrollPositionChanged");
			translateYPoppyView(ScrollDirection);
		}
		//Log.d(TAG, "oldScrollPosition : " + oldScrollPosition + "newScrollPosition : " + newScrollPosition);
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

				Log.d(TAG, "animate view! translationY : " + translationY);
				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(translationY);//translationY
			}
		});
	}

//	private void translateYSearchView() {
//		mPoppyView.post(new Runnable() {
//
//			//@Override
//			public void run() {
//				if(mPoppyViewHeight <= 0) {
//					mPoppyViewHeight = mPoppyView.getHeight();
//				}
//
//				int translationY = 0;
//				switch (mPoppyViewPosition) {
//					case BOTTOM:
//						translationY = mScrollDirection == SCROLL_TO_TOP ? 0 : mPoppyViewHeight;
//						break;
//					case TOP:
//						translationY = mScrollDirection == SCROLL_TO_TOP ? -mPoppyViewHeight : 0;
//						break;
//				}
//
//				ViewPropertyAnimator.animate(mPoppyView).setDuration(300).translationY(translationY);
//			}
//		});
//	}

	// for ListView

	int beforeCount = 0;
	private void initPoppyViewOnListView(EditText editView, ListView listView, final OnScrollListener onScrollListener) {
		setPoppyViewOnView(editView);
		Log.d(TAG,"editView.getheight : " + editView.getHeight());
		listView.setOnScrollListener(new OnScrollListener() {


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
				if(totalItemCount >= 5 && (firstVisibleItem + visibleItemCount) ==  totalItemCount)
				{
					//if(volumePage.CurrentPage < volumePage.totalPage)

					//if(!volumePage.bottom)//bottom이 제대로 될까..
					if(beforeCount != totalItemCount)//bottom이 제대로 될까..
					{
						Log.i("Info", "Scroll Bottom" );
						volumePage.getPage();
					}
					beforeCount = totalItemCount;
					//Log.i("Info", "Scroll Bottom" );
				}

//				Log.i("Info", "Height...newScrollPosition : " + newScrollPosition );
//				if(topChild != null) {
//					//Log.d(TAG, "lastest item's bottom : " + view.getChildAt(totalItemCount-1).getBottom());
//					Log.d(TAG, "view's Height : " + view.getHeight());
//					Log.d(TAG, "child's Height : " + view.getChildAt(0).getHeight());
//					Log.d(TAG, "Height..totalItemCount : " + totalItemCount);
//					Log.d(TAG, "calc total Height" + totalItemCount * view.getChildAt(0).getHeight());
//
//				}
//
//				if(topChild != null)
//				{
//					if((view.getHeight() - newScrollPosition) < 100 && (view.getHeight() - newScrollPosition) > 20) {
//						Log.i("Info", "Scroll Bottom");
//						if(!volumePage.bottom)
//						{
//							Log.d(TAG,"getPage, bottom : " + volumePage.bottom);
//							volumePage.bottom = true;
//							volumePage.getPage();
//						}
//					}
//					else
//					{
//						//bottom = false;
//					}
//				}
//
//				if (listView.getLastVisiblePosition() == listView.getAdapter().getCount() - 1
//						&& listView.getChildAt(listView.getChildCount() - 1).getBottom() <= list.getHeight()) {
//
//					Log.i("Info", "Scroll Bottom" );
//				}

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
