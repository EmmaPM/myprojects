from django.shortcuts import render, redirect, HttpResponseRedirect
from .forms import RegisterForm
from users.models import Resulttable, Insertposter,collection
import pymysql
import numpy as np
import random
import requests


def register(request):
    # 只有当请求为 POST 时，才表示用户提交了注册信息
    if request.method == 'POST':
        form = RegisterForm(request.POST)

        # 验证数据的合法性
        if form.is_valid():
            # 如果提交数据合法，调用表单的 save 方法将用户数据保存到数据库
            form.save()

            # 注册成功，跳转回首页
            return redirect('/')
    else:
        # 请求不是 POST，表明用户正在访问注册页面，展示一个空的注册表单给用户
        form = RegisterForm()

    # 渲染模板
    # 如果用户正在访问注册页面，则渲染的是一个空的注册表单
    # 如果用户通过表单提交注册信息，但是数据验证不合法，则渲染的是一个带有错误信息的表单
    return render(request, 'users/register.html', context={'form': form})


def index(request):
    return render(request, 'users/..//index.html')
    
 
def new_html(request):
    return render(request, 'users/new_html.html')


def check(request):
    return render((request, 'users/..//index.html'))


def showmessage(request):
    usermovieid = []
    #usermovietitle = []
    likemovieid=[]
    likemovietitle=[]
    ratings=[]      #用户对电影的所有评分
    nameANDrate = []
    data=Resulttable.objects.filter(userId=USERID)
    data2 = collection.objects.filter(userId=USERID)
    for row2 in data2:
        likemovieid.append(row2.imdbId)
        #print(likemovieid)
    for row in data:
        usermovieid.append(row.imdbId)
        #usermovieid.append(row.rating)
        #nameANDrate.append(usermovieid)

    try:
        conn = get_conn()
        cur = conn.cursor()
        conn2 = get_conn()
        curr = conn2.cursor()
        #Insertposter.objects.filter(userId=USERID).delete()
        for j in likemovieid:
            curr.execute('select * from moviegenre3 where imdbId = %s',j)
            rr2 = curr.fetchall()
            #print(rr2)
            for imdbId,title,poster in rr2:
                likemovietitle.append(title)        #得到每个用户收藏的电影名
               # print(title,imdbId)



        for i in usermovieid:
            cur.execute('select * from moviegenre3 where imdbId = %s',i)
            rr = cur.fetchall()
            for imdbId,title,poster in rr:
                #usermovietitle.append(title)        #得到每一用户评分的电影名
                #print(title)
                ratings = Resulttable.objects.get(userId=USERID, imdbId=imdbId)
                # ratings.filter()
                #usermovietitle.append(ratings.rating)
                #print(ratings.rating)
                # print(likemovietitle)
                #usermovietitle.append(str(ratings.rating))
                Onerating = title + "——" + str(ratings.rating) + "分"
                nameANDrate.append(Onerating)
                #print(nameANDrate)

        # print(poster_result)
    finally:
        conn.close()
        conn2.close()
    return render(request, 'users/message1.html', locals())


def recommend(request):
    return render(request, 'users/movieRecommend.html')


def insert(request):
    # MOVIEID = int(request.GET["movieId"])
    global USERID
    USERID = int(request.GET["userId"])
    # USERID = {{}}
    RATING = float(request.GET["rating"])
    IMDBID = int(request.GET["imdbId"])
    row = Resulttable.objects.filter(userId=USERID)         #评分
    for data in row:
        if IMDBID == data.imdbId:
            Resulttable.objects.filter(imdbId=IMDBID).delete()
    Resulttable.objects.create(userId=USERID, rating=RATING,imdbId=IMDBID)

    #print(USERID)
    # return HttpResponseRedirect('/')
    return render(request, 'index.html',{'userId':USERID,'rating':RATING,'imdbId':IMDBID})

def islike(request):
    global USERID
    USERID = int(request.GET["userId1"])
    #print(USERID)
    # USERID = {{}}
    IMDBID = int(request.GET["loveId"])
    if collection.objects.filter(userId=USERID,imdbId=IMDBID).exists():
        collection.objects.filter(imdbId=IMDBID).delete()  # 已经有了就删除，表示取消收藏
    else:
        collection.objects.create(userId=USERID, imdbId=IMDBID, islike=True)
    #collection.objects.filter().delete()
    # for row in rows:
    #     if IMDBID == row.imdbId:
    #         collection.objects.filter(imdbId=IMDBID).delete()       #已经有了就删除，表示取消收藏
    #     else:
    #         collection.objects.create(userId=USERID,imdbId=IMDBID,islike=True)      #收藏
     #collection.objects.filter().delete()
    return render(request, 'index.html',{'userId':USERID,'loveId':IMDBID})


def get_conn():
    conn = pymysql.connect(host='139.155.124.92', port=3306, user='test_2', passwd='666', db='test_2', charset='utf8')
    return conn


def query_all(cur, sql, args):
    cur.execute(sql, args)
    return cur.fetchall()
   

class MyALS:
    def __init__(self, user_ids, item_ids, ratings, rank, iterations=5, lambda_=0.01, blocks=-1, seed=None):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.rank = rank
        self.iterations = iterations
        self.lambda_ = lambda_
        self.blocks = blocks
        self.seed = seed
        self.user_matrix = None
        self.item_matrix = None
        self.rmse = None
        self.ratings_size = ratings.shape
        self.user_dict = dict(zip(user_ids, range(len(user_ids))))
        self.item_dict = dict(zip(item_ids, range(len(item_ids))))

    def _preprocess(self):
        user_n = len(self.user_ids)
        item_n = len(self.item_ids)
        if self.ratings_size != (user_n, item_n):
            print("matrix ratings must be suitable with user_ids and iten_ids")
            raise IndexError
        if self.rank > item_n:
            print("rank must less than item number")
            raise IndexError

    def _random_matrix(self):
        np.random.seed(self.seed)
        self.user_matrix = np.random.rand(self.ratings_size[0], self.rank)

    def _get_rmse(self):
        """" rmse = sqrt(sum[(ratings - premat)^2]/N) """
        predict_matrix = np.matmul(self.user_matrix, self.item_matrix)
        de = np.array(self.ratings - predict_matrix)
        self.rmse = (sum(sum(de ** 2)) / (self.ratings_size[0] * self.ratings_size[1])) ** 0.5

    def _get_item_matrix(self):
        """ Y = (X^T*X + lambda*I)^-1*X^T*ratings """
        self.item_matrix = np.matmul(
            np.matmul((np.linalg.pinv(np.matmul(self.user_matrix.T, self.user_matrix) + self.lambda_)),
                      self.user_matrix.T), self.ratings)

    def _get_user_matrix(self):
        """ X = ((Y*Y^T + lambda*I)^-1*Y*ratings^T)^T """
        self.user_matrix = np.matmul(
            np.matmul((np.linalg.pinv(np.matmul(self.item_matrix, self.item_matrix.T) + self.lambda_)),
                      self.item_matrix), self.ratings.T).T

    def learn_para(self, rankrange, iterationrange, lambdarange):
        self._preprocess()
        para = dict()
        for self.rank in range(rankrange[0], rankrange[1], rankrange[2]):
            for self.iterations in range(iterationrange[0], iterationrange[1], iterationrange[2]):
                for self.lambda_ in np.arange(lambdarange[0], lambdarange[1], lambdarange[2]):
                    print("para <rank, iteration, lambda>", self.rank, self.iterations, self.lambda_)
                    self._random_matrix()
                    self._get_item_matrix()
                    self._get_user_matrix()
                    self._get_rmse()
                    firstrmse = self.rmse
                    for k in range(self.iterations - 1):
                        self._get_item_matrix()
                        self._get_user_matrix()
                        self._get_rmse()
                    if self.rmse < firstrmse:
                        para[self.rank, self.iterations, self.lambda_] = self.rmse
                        print("converge, rmse: ", self.rmse)
        return para

    def fit(self):
        self._preprocess()
        self._random_matrix()
        for k in range(self.iterations):
            self._get_item_matrix()
            self._get_user_matrix()
            self._get_rmse()
            print("Iterations: {0}, RMSE: {1:.6}".format(k + 1, self.rmse))

    def predict(self, user_id, n_items=10):
        if type(user_id) is not list:
            user_id = [user_id]
        k = []
        predict_user_matrix = np.zeros((len(user_id), self.rank))
        for m in range(len(user_id)):
            k.append(self.user_ids.index(user_id[m]))
            predict_user_matrix[m] = self.user_matrix[k[m]]
        scores_matrix = np.matmul(predict_user_matrix, self.item_matrix)
        scores_dict = dict()
        t = 0
        for id1 in user_id:
            scores_dict[id1] = dict(zip(self.item_ids, scores_matrix.tolist()[t]))
            for item in self.item_ids:
                if self.ratings[self.user_dict[id1], self.item_dict[item]] != 0:
                    del scores_dict[id1][item]
            t += 1
        recc = dict()
        for id1 in user_id:
            recc[id1] = sorted(scores_dict[id1].items(), key=lambda x: x[1], reverse=True)[:n_items]
        return recc


if __name__ == "__main__":
    print("test ALS...")
    """
    user_ids = [1, 22, 333, 4444]
    item_ids = [111, 222, 333, 444, 555, 666, 777]
    ratings = np.matrix([[0, 4, 3, 5, 0, 0, 1],
                        [2, 0, 5, 1, 0, 0, 3],
                        [5, 3, 2, 1, 3, 5, 0],
                        [1, 2, 3, 4, 5, 3, 2]])
    """
    user_ids = random.sample(range(1, 1000), 15)
    item_ids = random.sample(range(1, 100000), 40000)
    ratings = np.random.randint(0, 5, (15, 40000))
    # """

    model = MyALS(user_ids, item_ids, ratings, rank=10, iterations=5, lambda_=0.1)
    model.fit()
    rec = model.predict(user_ids[:3], 3)
    print(rec)
    # model = MyALS(user_ids, item_ids, ratings, rank=1, seed=1)
    # para = model.learn_para([1, 100, 10], [5, 20, 5], [0.01, 0.5, 0.04])
    # print(para)

USERID = 0


def recommend3(request):
    global USERID
    try:
        USERID = int(request.GET["userIddd"])
    except:
        return redirect('http://139.155.124.92:8000/users/login')
        
    Insertposter.objects.filter(userId=USERID).delete()

    conn = pymysql.connect(host='139.155.124.92', port=3306, user='test_2', passwd='666', db='test_2', charset='utf8')
    cur = conn.cursor()
    cur.execute('select imdbId from moviegenre3')
    movie_ids = cur.fetchall()
    movie_ids = [k[0] for k in movie_ids]
    random.shuffle(movie_ids)

    cur.execute('select * from users_resulttable')
    user_r = cur.fetchall()
    user_ids = []
    for k in user_r:
        if k[1] not in user_ids:
            user_ids.append(k[1])
    print('USERID:', USERID, '\n', 'user_ids: ', user_ids)
    
    if USERID not in user_ids:
        return redirect('http://139.155.124.92:8000')
    
    ratings = np.zeros(shape=(len(user_ids), len(movie_ids)))
    for row_user in user_r:
        uid, mid, score = row_user[1:]
        if uid in user_ids:
            if mid in movie_ids:
                ratings[user_ids.index(uid)][movie_ids.index(mid)] = float(score)

    print(len(user_ids), len(movie_ids), ratings.shape)
    model = MyALS(user_ids, movie_ids, ratings, rank=7, iterations=5, lambda_=0.5)
    model.fit()
    recc = model.predict(USERID, 10)
    print(recc)

    try:
        for rec in recc[USERID]:
            cur.execute('select * from moviegenre3 where imdbId = %d' % rec[0])
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                Insertposter.objects.create(userId=USERID, title=title, poster=poster)
    finally:
        conn.close()
    results = Insertposter.objects.filter(userId=USERID)
    return render(request, 'users/movieRecommend.html', locals())
    
def switch(request):
    return render(request, 'users/switch.html')   


