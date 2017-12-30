# hait-task1{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 実践課題 vol.1\n",
    "AI_STANDARD 機械学習講座vol.1~ vol.5の学習のアウトプットとして、実践的な課題を解いて行きましょう。kaggleというデータサイエンスのコンペティションサイトにある、住宅価格予測のデータを使用した課題です。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "![](https://s3-ap-northeast-1.amazonaws.com/ai-std/kadai1-1.png)\n",
    "\n",
    "[kaggle: House Prices Competition link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## データのダウンロード\n",
    "２つのデータをダウンロードして、jupyterを起動しているディレクトリに保存して下さい(右クリックでダウンロードを選択できます)。\n",
    "\n",
    " - <a href=\"https://s3-ap-northeast-1.amazonaws.com/ai-std/house_price.csv\" download=\"house_price.csv\">前処理済みデータ</a>\n",
    " - <a href=\"https://s3-ap-northeast-1.amazonaws.com/ai-std/y.csv\" download=\"y.csv\">正解データ</a>\n",
    " \n",
    " ![](https://s3-ap-northeast-1.amazonaws.com/ai-std/kadai1-2.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# モジュールのインポート\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "% matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題１：dataの上から20行を表示してください。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    LotFrontage   LotArea    Street  LotShape  Utilities  LandSlope  \\\n",
      "0            65  9.042040  1.098612  1.609438   1.609438   1.386294   \n",
      "1            80  9.169623  1.098612  1.609438   1.609438   1.386294   \n",
      "2            68  9.328212  1.098612  1.386294   1.609438   1.386294   \n",
      "3            60  9.164401  1.098612  1.386294   1.609438   1.386294   \n",
      "4            84  9.565284  1.098612  1.386294   1.609438   1.386294   \n",
      "5            85  9.555064  1.098612  1.386294   1.609438   1.386294   \n",
      "6            75  9.218804  1.098612  1.609438   1.609438   1.386294   \n",
      "7             0  9.247925  1.098612  1.386294   1.609438   1.386294   \n",
      "8            51  8.719481  1.098612  1.609438   1.609438   1.386294   \n",
      "9            50  8.912069  1.098612  1.609438   1.609438   1.386294   \n",
      "10           70  9.323758  1.098612  1.609438   1.609438   1.386294   \n",
      "11           85  9.386392  1.098612  1.386294   1.609438   1.386294   \n",
      "12            0  9.470317  1.098612  1.098612   1.609438   1.386294   \n",
      "13           91  9.273597  1.098612  1.386294   1.609438   1.386294   \n",
      "14            0  9.298443  1.098612  1.386294   1.609438   1.386294   \n",
      "15           51  8.719481  1.098612  1.609438   1.609438   1.386294   \n",
      "16            0  9.327412  1.098612  1.386294   1.609438   1.386294   \n",
      "17           72  9.286560  1.098612  1.609438   1.609438   1.386294   \n",
      "18           66  9.524859  1.098612  1.609438   1.609438   1.386294   \n",
      "19           70  8.930759  1.098612  1.609438   1.609438   1.386294   \n",
      "\n",
      "    OverallQual  OverallCond  YearBuilt  YearRemodAdd          ...            \\\n",
      "0             7     1.791759   7.602900      7.602900          ...             \n",
      "1             6     2.197225   7.589336      7.589336          ...             \n",
      "2             7     1.791759   7.601902      7.602401          ...             \n",
      "3             7     1.791759   7.557995      7.586296          ...             \n",
      "4             8     1.791759   7.601402      7.601402          ...             \n",
      "5             5     1.791759   7.597898      7.598900          ...             \n",
      "6             8     1.791759   7.603399      7.603898          ...             \n",
      "7             7     1.945910   7.587817      7.587817          ...             \n",
      "8             7     1.791759   7.566311      7.576097          ...             \n",
      "9             5     1.945910   7.570443      7.576097          ...             \n",
      "10            5     1.791759   7.583756      7.583756          ...             \n",
      "11            9     1.791759   7.603898      7.604396          ...             \n",
      "12            5     1.945910   7.582229      7.582229          ...             \n",
      "13            7     1.791759   7.604396      7.604894          ...             \n",
      "14            6     1.791759   7.581210      7.581210          ...             \n",
      "15            7     2.197225   7.565275      7.601902          ...             \n",
      "16            6     2.079442   7.586296      7.586296          ...             \n",
      "17            4     1.791759   7.584773      7.584773          ...             \n",
      "18            5     1.791759   7.603399      7.603399          ...             \n",
      "19            5     1.945910   7.580189      7.583756          ...             \n",
      "\n",
      "    SaleType_ConLw  SaleType_New  SaleType_Oth  SaleType_WD  \\\n",
      "0                0             0             0            1   \n",
      "1                0             0             0            1   \n",
      "2                0             0             0            1   \n",
      "3                0             0             0            1   \n",
      "4                0             0             0            1   \n",
      "5                0             0             0            1   \n",
      "6                0             0             0            1   \n",
      "7                0             0             0            1   \n",
      "8                0             0             0            1   \n",
      "9                0             0             0            1   \n",
      "10               0             0             0            1   \n",
      "11               0             1             0            0   \n",
      "12               0             0             0            1   \n",
      "13               0             1             0            0   \n",
      "14               0             0             0            1   \n",
      "15               0             0             0            1   \n",
      "16               0             0             0            1   \n",
      "17               0             0             0            1   \n",
      "18               0             0             0            1   \n",
      "19               0             0             0            0   \n",
      "\n",
      "    SaleCondition_Abnorml  SaleCondition_AdjLand  SaleCondition_Alloca  \\\n",
      "0                       0                      0                     0   \n",
      "1                       0                      0                     0   \n",
      "2                       0                      0                     0   \n",
      "3                       1                      0                     0   \n",
      "4                       0                      0                     0   \n",
      "5                       0                      0                     0   \n",
      "6                       0                      0                     0   \n",
      "7                       0                      0                     0   \n",
      "8                       1                      0                     0   \n",
      "9                       0                      0                     0   \n",
      "10                      0                      0                     0   \n",
      "11                      0                      0                     0   \n",
      "12                      0                      0                     0   \n",
      "13                      0                      0                     0   \n",
      "14                      0                      0                     0   \n",
      "15                      0                      0                     0   \n",
      "16                      0                      0                     0   \n",
      "17                      0                      0                     0   \n",
      "18                      0                      0                     0   \n",
      "19                      1                      0                     0   \n",
      "\n",
      "    SaleCondition_Family  SaleCondition_Normal  SaleCondition_Partial  \n",
      "0                      0                     1                      0  \n",
      "1                      0                     1                      0  \n",
      "2                      0                     1                      0  \n",
      "3                      0                     0                      0  \n",
      "4                      0                     1                      0  \n",
      "5                      0                     1                      0  \n",
      "6                      0                     1                      0  \n",
      "7                      0                     1                      0  \n",
      "8                      0                     0                      0  \n",
      "9                      0                     1                      0  \n",
      "10                     0                     1                      0  \n",
      "11                     0                     0                      1  \n",
      "12                     0                     1                      0  \n",
      "13                     0                     0                      1  \n",
      "14                     0                     1                      0  \n",
      "15                     0                     1                      0  \n",
      "16                     0                     1                      0  \n",
      "17                     0                     1                      0  \n",
      "18                     0                     1                      0  \n",
      "19                     0                     0                      0  \n",
      "\n",
      "[20 rows x 290 columns]\n",
      "         y\n",
      "0   208500\n",
      "1   181500\n",
      "2   223500\n",
      "3   140000\n",
      "4   250000\n",
      "5   143000\n",
      "6   307000\n",
      "7   200000\n",
      "8   129900\n",
      "9   118000\n",
      "10  129500\n",
      "11  345000\n",
      "12  144000\n",
      "13  279500\n",
      "14  157000\n",
      "15  132000\n",
      "16  149000\n",
      "17   90000\n",
      "18  159000\n",
      "19  139000\n"
     ]
    }
   ],
   "source": [
    "X = pd.read_csv('../Downloads/house_price.csv')\n",
    "y = pd.read_csv('../Downloads/y.csv')\n",
    "\n",
    "print(X.head(20))\n",
    "print(y.head(20))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題２：ホールド・アウト法によるデータの分割をしてください。\n",
    "条件：テストデータの割合は3割、random_stateは0、変数は「X_train, X_test, y_train, y_test」を使用"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題３：線形回帰モデルを作成してください。\n",
    " - モジュールのインポート\n",
    " - インスタンスの生成\n",
    " - モデルへのfit\n",
    " - scoreの表示"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr=LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train:0.930\n",
      "test:0.692\n"
     ]
    }
   ],
   "source": [
    "print('train:%.3f'%lr.score(X_train,y_train))\n",
    "print('test:%.3f'%lr.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題４：Ridge回帰モデルを作成してください。\n",
    " - モジュールのインポート\n",
    " - インスタンスの生成（引数：alpha=10）\n",
    " - モデルへのfit\n",
    " - テストデータでのscoreの表示"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import Ridge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "mr = Ridge(alpha=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None,\n",
       "   normalize=False, random_state=None, solver='auto', tol=0.001)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mr.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test:0.809\n"
     ]
    }
   ],
   "source": [
    "print('test:%.3f'%mr.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題５：LASSOモデルを作成してください。\n",
    " - モジュールのインポート\n",
    " - インスタンスの生成（引数：alpha=200）\n",
    " - モデルへのfit\n",
    " - テストデータでのscoreの表示"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import Lasso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "lm=Lasso(alpha=200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lasso(alpha=200, copy_X=True, fit_intercept=True, max_iter=1000,\n",
       "   normalize=False, positive=False, precompute=False, random_state=None,\n",
       "   selection='cyclic', tol=0.0001, warm_start=False)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lm.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test:0.808\n"
     ]
    }
   ],
   "source": [
    "print('test:%.3f'%lm.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題６：Elastic Netモデルを作成してください。\n",
    " - モジュールのインポート\n",
    " - インスタンスの生成（引数：alpha=0.1, l1_ratio=0.9）\n",
    " - モデルへのfit\n",
    " - scoreの表示"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import ElasticNet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "enm=ElasticNet(alpha=0.1,l1_ratio=0.9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/yumaikeda/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.\n",
      "  ConvergenceWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.9,\n",
       "      max_iter=1000, normalize=False, positive=False, precompute=False,\n",
       "      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "enm.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train:0.889\n",
      "test:0.807\n"
     ]
    }
   ],
   "source": [
    "print('train:%.3f'%enm.score(X_train,y_train))\n",
    "print('test:%.3f'%enm.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
