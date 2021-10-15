

import numpy as np
import pandas as pd
from multiprocessing import Pool, Process

from scipy.spatial.distance import cosine

print(__name__) 

def write_cos_dist(t):
    print(t[0].shape)
    print(t[1].shape)
    res = t[0].apply(cosine, axis=1, args=(t[1],))
    return res


if __name__ == '__main__':

    from collections import Counter
    import math
    import re
    import model as model_py
    import threading
    import multiprocessing
    import math

    user = {}

    def initializer():
       multiprocessing.current_process().name = 'helper'
       threading.current_thread().name = 'helper'

    class MusicRecommender():
        """
        Takes as input some Yandex Music track ids and gives 
        as output id of tracks you should probably listen to!
        
        Requirements:
            numpy
            pandas
            regex
            scipy.spatial.distance.cosine
            collections.Counter
            math
        """
        def __init__(self, df_songs, df_song_names, df_users, df_users_preferences):
            self.df_songs = df_songs#pd.read_csv('songs_dataset_transformed.csv', ',', index_col='track_id', encoding='cp1251')
            self.df_song_names = df_song_names#pd.read_csv('songs_dataset.csv',';', encoding='cp1251', index_col='track_id').drop_duplicates()
            self.df_users = df_users#pd.read_csv('users_dataset.csv', ';', encoding='cp1251')
            self.df_users_preferences = df_users_preferences#pd.read_csv('users_preferences_dataset.csv', index_col='user_id')
            self.users_track_id_valid = [id_ for id_ in self.df_users['track_id'] if id_ in self.df_song_names.index]
            self.songs_model_weights_coldstart = self.coldstart_model()
            self.neighbors_users_count = 25
            return None

        def take_average_footprint_by_songs_ids(self, ids, consider_relevance=False):
            
            ids = [int(id) for id in ids if int(id) in df_songs.index]
            print(ids)
            how_many_songs = len(ids)
            if how_many_songs > 0:
                feature_list = self.df_songs.columns
                user_cumulative_info = pd.Series({feature:0 for feature in feature_list})

                if consider_relevance:
                    ordinal_coefficients = {i:self.song_time_relevance_sigmoid(i) for i in range(1, how_many_songs+1)}
                    norma_coef = sum(ordinal_coefficients.values())
                    for key,value in ordinal_coefficients.items():
                        ordinal_coefficients[key] = value/norma_coef 

                curr_order = 1
                for track_id in ids:
                    try:
                        if consider_relevance == False:
                            print(self.df_songs.loc[track_id])
                            user_cumulative_info += self.df_songs.loc[track_id]
                        else:
                            print(self.df_songs.loc[track_id])
                            user_cumulative_info += self.df_songs.loc[track_id]*ordinal_coefficients[curr_order]
                    except Exception as e:
                        print(e)
                        how_many_songs -= 1
                    curr_order += 1

                if not consider_relevance:
                    user_cumulative_info /= how_many_songs
                else:
                    user_cumulative_info *= len(ids)/how_many_songs

                genre_filter = re.compile('genre_[a-z]*')
                genre_cols = [col for col in feature_list if genre_filter.match(col)]
                user_cumulative_info[genre_cols] /= max(user_cumulative_info[genre_cols])
                user_cumulative_info[genre_cols] *=2

                return user_cumulative_info
            else:
                return self.take_average_footprint_by_songs_ids(df_users_preferences.dropna().index, False)
        
        def coldstart_coeff_sigmoid(self, n_songs):
            if n_songs < 15:
                hard_factor = -1/2
                offset = 6.5
                return 1 - (1/(1+math.exp(hard_factor*(n_songs-offset))) - 1/(1+math.exp(hard_factor*(-offset))))
            else:
                return 0
        
        def song_time_relevance_sigmoid(self, x):
            if x > 15:
                hard_factor = 1/10
                offset = 40
                return 1/(1+math.exp(hard_factor*(x-offset))) + 0.075
            else:
                return 1.
        
        def str_int_list_to_list(self, str_):
            no_brackets = str_.replace(']', '').replace('[', '').replace(' ', '')
            if len(no_brackets) > 0:
                return([int(t) for t in no_brackets.split(',')])
            else:
                return []
        
        def coldstart_model(self):
            most_popular_artists = self.df_song_names.loc[self.users_track_id_valid]['artist'].value_counts()[2:30]
            most_popular_artists_songs = self.df_song_names[self.df_song_names['artist'].isin(dict(most_popular_artists).keys())]

            bests_from_popular_artists = []

            for index,row in most_popular_artists_songs.iterrows():
                bests_from_popular_artists.extend(self.str_int_list_to_list(row['bests_from_album']))

            bests_from_popular_artists = [track_id for track_id in list(set(bests_from_popular_artists))
                                        if track_id in self.df_song_names.index]
            bests_from_popular_artists_df_sorted = self.df_song_names.loc[bests_from_popular_artists]
            bests_from_popular_artists_df_sorted = bests_from_popular_artists_df_sorted[
                (bests_from_popular_artists_df_sorted['likes_count'] != 'None') &
                (bests_from_popular_artists_df_sorted['duration_ms'] > 120000)]
            bests_from_popular_artists_df_sorted = bests_from_popular_artists_df_sorted.drop_duplicates()
            bests_from_popular_artists_df_sorted['likes_count'] = bests_from_popular_artists_df_sorted['likes_count'].astype(int)

            bests_all_time = open('../data/bests.csv', 'r')
            bests_all_time_l = [int(s) for s in bests_all_time.readline().replace(' ', '').split(',')]
            bests_all_time.close()

            bests_all_time_df = self.df_song_names.loc[bests_all_time_l]

            songs_to_boost = pd.concat([bests_from_popular_artists_df_sorted.sort_values('likes_count', ascending=False)[:350], 
                                        bests_all_time_df]).sample(25, random_state=100)

            model_track_weights = pd.Series({track_id:0 for track_id in self.df_song_names.index})

            for best_track_id in songs_to_boost.index:
                model_track_weights[best_track_id] = 0.85
            return model_track_weights
        
        def add_neighbor_tracks(self, neighbor_id):
            track_indecies = [id_ 
                            for id_ in self.df_users[self.df_users['owner_id'] == neighbor_id]['track_id'] 
                            if id_ in self.df_songs.index]
            self.songs_neighbors.extend(track_indecies)
        
        def user_based_distanse_decreasing_sigmoid(self, x):
            """
            Takes percent of users that have certain song and returns
            coefficient for cosine distance to decrease.
            """
            hard_factor = -7
            offset = 0.4
            return 1/(1+math.exp(hard_factor*(x-offset))) - 1/(1+math.exp(hard_factor*(-offset)))
        
        def user_based_model(self, list_user_tracks_id):
            global user
            user = self.take_average_footprint_by_songs_ids(list_user_tracks_id, consider_relevance=True)
            print(self.df_users_preferences.shape, len(user))
            cosine_dists = self.df_users_preferences.apply(cosine, axis=1, args=(user,))

            cosine_dists_sorted = sorted(list(cosine_dists.items()), key=lambda x:x[1])

            users_neighbors_to_consider = [(user_dist[0], 1-user_dist[1])
                                        for user_dist in cosine_dists_sorted[:self.neighbors_users_count]]
            users_neighbors = [t[0] for t in users_neighbors_to_consider]

            self.songs_neighbors = []
            add_neighbor_tracks_v = np.vectorize(self.add_neighbor_tracks)
            add_neighbor_tracks_v(users_neighbors)

            relevant_artists = self.df_song_names.loc[self.songs_neighbors]['artist'].value_counts(normalize=True).head(50)

            relevant_artists_normalized = relevant_artists/(relevant_artists[0]/self.artists_importance)

            #relevant_genres = self.df_song_names.loc[self.songs_neighbors]['genre'].value_counts(normalize=True).head(5)
            #relevant_genres = relevant_genres/(relevant_genres[0]/self.genres_importance)
            relevant_genres = pd.Series({genre_chosen:self.genres_importance for genre_chosen in self.genres_chosen})
            
            all_sim_songs = []

            for track_id_lst in self.df_song_names.loc[[track_id for track_id in list_user_tracks_id
                                                    if track_id in self.df_songs.index]]['bests_from_album']:
                no_brackets = track_id_lst.replace(']', '').replace('[', '').replace(' ', '')
                if len(no_brackets) > 0:
                    all_sim_songs.extend([int(t) for t in no_brackets.split(',')])

            all_sim_songs = set(all_sim_songs)
            
            self.songs_model_weights_user_based = pd.Series({sim_song:self.sim_songs_importance for sim_song in all_sim_songs})
            self.genres_model_weights = relevant_genres
            self.artists_model_weights = relevant_artists_normalized
        
        def artists_vs_weight_into_songs_vs_weights(self, dict_artists_vs_weights):
            result = {}
            for artist, weight in dict_artists_vs_weights.items():
                for track_id in self.df_song_names[self.df_song_names['artist'] == artist].index:
                    result[track_id] = weight
            return pd.Series(result)
        
        def genres_vs_weight_into_songs_vs_weights(self, dict_genres_vs_weights):
            result = {}
            for genre, weight in dict_genres_vs_weights.items():
                for track_id in self.df_song_names[self.df_song_names['genre'] == genre].index:
                    result[track_id] = weight
            return pd.Series(result)
        
        def apply_song_weights(self, dict_song_ids_vs_weights, coeff):
            indecies_to_update = dict_song_ids_vs_weights.index

            self.result_weights.loc[indecies_to_update] =  1 - (1 - self.result_weights.loc[indecies_to_update])* \
                                                               (1 - dict_song_ids_vs_weights*coeff)
            #for track_id, weight in dict_song_ids_vs_weights.items():
            #    self.result_weights[track_id] = 1. - (1. - self.result_weights[track_id])*(1.-1.*weight*coeff)
            pass
        
        def fit(self, genres_chosen, artists_chosen, tracks_chosen, 
                learning_rate=0.05, 
                artists_importance=0.05, 
                genres_importance=0.30,
                sim_songs_importance=0.1,
                sim_artists_importance=0.45):
            
            self.artists_importance = artists_importance
            self.genres_importance = genres_importance
            self.sim_songs_importance = sim_songs_importance
            self.sim_artists_importance = sim_artists_importance
            self.tracks_chosen = tracks_chosen
            self.genres_chosen = genres_chosen
            if len(tracks_chosen) > 0:
                print(tracks_chosen)
                self.user_based_model(tracks_chosen)
                    
            coldstart_coeff = self.coldstart_coeff_sigmoid(len(tracks_chosen))
            
            self.result_weights = pd.Series({track_id:0 for track_id in self.df_song_names.index})
            
            all_sim_artists = []
            for artist_name in artists_chosen:
                similar_artists = pd.Series(
                        dict(
                            Counter(
                                df_users[
                                    df_users['owner_id'].isin(df_users[df_users['artist_name'] == artist_name]['owner_id'].unique())]
                                ['artist_name'])).items()).sort_values(ascending=False)
                try:
                    similar_artists.remove(artist_name)
                except Exception:
                    pass
                similar_artists = similar_artists[:5]
                all_sim_artists.extend(similar_artists)

            self.artists_model_weights_user_based = pd.Series({
                artist_name:self.sim_artists_importance for artist_name in all_sim_artists})

            self.apply_song_weights(self.songs_model_weights_coldstart, coldstart_coeff)
            self.apply_song_weights(self.songs_model_weights_user_based, 1. - coldstart_coeff)
            self.apply_song_weights(self.genres_vs_weight_into_songs_vs_weights(self.genres_model_weights), 
                                    1. - coldstart_coeff)
            self.apply_song_weights(self.artists_vs_weight_into_songs_vs_weights(self.artists_model_weights), 
                                    1. - coldstart_coeff)
            self.apply_song_weights(self.artists_vs_weight_into_songs_vs_weights(self.artists_model_weights_user_based), 
                                    1. - coldstart_coeff)

            print(self.artists_model_weights_user_based)
            print(self.artists_model_weights)
            print(self.genres_model_weights)
            pass
        
        def update_dists(self, track_id_vs_coeff):
            try:
                self.all_cos_distances[track_id_vs_coeff[0]] = self.all_cos_distances[track_id_vs_coeff[0]]*(1.-track_id_vs_coeff[1])
            except Exception:
                pass

        def predict(self, predict_count=20):
            print('cos_dist_start')
            
            global user

            num_workers = 8
            pool = Pool(num_workers)
            len_df = recommender.df_songs.shape[0]
            self.all_cos_distances = pd.concat(pool.map(write_cos_dist, [(recommender.df_songs[int(len_df*i/num_workers):int(len_df*(i+1)/num_workers)], user) for i in range(num_workers)]))
            pool.close()
            pool.join()

            print('cos_dist_end')

            rw = pd.Series(self.result_weights)
            rw = rw[rw > 0]
            cd = pd.Series(self.all_cos_distances).dropna()
            cd = cd.loc[[ind for ind in rw.index if ind in cd.index]]
            updated_dists = cd * (1 - rw)
            self.all_cos_distances = self.all_cos_distances.drop(self.all_cos_distances.loc[self.all_cos_distances.index.duplicated()].index)
            updated_dists = updated_dists.drop(updated_dists.loc[updated_dists.index.duplicated()].index)
            indecies_to_apply = [ind for ind in updated_dists.index if ind in self.all_cos_distances.index]
            self.all_cos_distances.loc[indecies_to_apply] = updated_dists.loc[indecies_to_apply]            

            track_ids_sorted = self.all_cos_distances.sort_values()
            
            for track_id_already_exist in self.tracks_chosen:
                try:
                    track_ids_sorted.remove(track_id_already_exist)
                except Exception:
                    pass
            df_to_return = self.df_song_names.loc[track_ids_sorted.index][:predict_count*5].sample(predict_count)  
            print('RMSE:', math.sqrt(sum(track_ids_sorted.loc[df_to_return.index].values ** 2)))
            return df_to_return
    

    df_songs = model_py.df_songs #pd.read_csv('songs_dataset_transformed.csv', ',', index_col='track_id', encoding='cp1251')
    df_song_names = model_py.df_song_names#pd.read_csv('songs_dataset.csv',';', encoding='cp1251', index_col='track_id').drop_duplicates()
    df_users = model_py.df_users#pd.read_csv('users_dataset.csv', ';', encoding='cp1251')
    df_users_preferences = model_py.df_users_preferences#pd.read_csv('users_preferences_dataset.csv', index_col='user_id')
    
    recommender = MusicRecommender(df_songs, df_song_names, df_users, df_users_preferences)
    def get_recommends(artists_like, genres_like, tracks_like):
        print('fit')
        recommender.fit(genres_like, artists_like, tracks_like)
        print('predict')
        pred = recommender.predict()
        print('predicted')
        print(pred)
        pred_json = {'tracks': [{'track_id': index,
                                    'album_id': row['album_id'],
                                    'artist_id': row['artist_id'],
                                    'artist': row['artist'],
                                    'title': row['title'],
                                    'album': row['album'],
                                    'song_version': row['song_version'],
                                    'duration': row['duration_ms']
                                    } for index, row in pred[['album_id', 'artist_id', 'artist', 'title', 'album', 'song_version', 'duration_ms']].iterrows()]}
        return pred_json


    from flask import Flask, render_template, request, url_for, request, redirect, abort
    from flask_login import LoginManager, login_user, logout_user, login_required, current_user
    from flask_talisman import Talisman
    from flask_pymongo import PyMongo
    from flask_bcrypt import Bcrypt
    from flask_wtf.csrf import CSRFProtect

    # Other modules
    from urllib.parse import urlparse, urljoin
    from datetime import datetime
    import configparser
    import json
    import sys
    import os

    # Local imports
    from user import User, Anonymous
    from verification import confirm_token
    import model as model_py
    import predict

    app = Flask(__name__)

    config = configparser.ConfigParser()
    config.read('configuration.ini')
    default = config['DEFAULT']
    app.secret_key = default['SECRET_KEY']
    app.config['MONGO_DBNAME'] = default['DATABASE_NAME']
    app.config['MONGO_URI'] = default['MONGO_URI']
    app.config['PREFERRED_URL_SCHEME'] = "https"

    mongo = PyMongo(app)

    bc = Bcrypt(app)

    csp = {
        'default-src': [
            '\'self\'',
            'https://stackpath.bootstrapcdn.com',
            'https://pro.fontawesome.com',
            'https://code.jquery.com',
            'https://cdnjs.cloudflare.com'
        ]
    }
    talisman = Talisman(app, content_security_policy=csp)

    csrf = CSRFProtect()
    csrf.init_app(app)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.anonymous_user = Anonymous
    login_manager.login_view = "login"

    # Index
    @app.route('/')
    @login_required
    def index():
        return render_template('index.html')

    # Login
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'GET':
            if current_user.is_authenticated:
                return redirect(url_for('index'))
            return render_template('login.html', error=request.args.get("error"))
        users = mongo.db.usersDB
        user_data = users.find_one({'login': request.form['login']}, {'_id': 0})
        if user_data:
            if bc.check_password_hash(user_data['password'], request.form['pass']):
                user_dict = {'login': user_data['login'],'id': user_data['id']}
                user = User.make_from_dict(user_dict)
                login_user(user)
                return redirect(url_for('index'))

        return redirect(url_for('login', error=1))

    @app.route('/get_genres', methods=['GET'])
    def get_genres():
        return model_py.genres   

    def str_list_to_list(lst_str):
        return lst_str.replace('[', '').replace(']', '').replace('\"', '').split(',')

    @app.route('/get_artists', methods=['GET'])
    def get_artists():   
        genres_lst = request.args.get('genres_lst')
        print(genres_lst)
        return model_py.return_artists(genres_lst)

    @app.route('/model')
    def model():
        users_prefs = mongo.db.users_preferencesDB
        existing_pref = users_prefs.find_one({'_id': current_user.id})
        current_user.genres_like = str_list_to_list(existing_pref['genres_like'])
        current_user.artists_like = str_list_to_list(existing_pref['artists_like'])
        return render_template('model.html')

    @app.route('/save_preferences', methods=['GET'])
    def save_preferences():
        genres_like = request.args.get('genres_lst')
        artists_like = request.args.get('artists_lst')
        users_prefs = mongo.db.users_preferencesDB
        existing_pref = users_prefs.find_one({'_id': current_user.id})
        if existing_pref is not None:
            users_prefs.update_one({'_id': current_user.id}, {
                '$set': {
                    'artists_like': artists_like,
                    'genres_like': genres_like
                }
            })
        else:
            users_prefs.insert_one({'_id': current_user.id,
                        'artists_like': artists_like,
                        'genres_like': genres_like
            })
        current_user.genres_like = str_list_to_list(genres_like)
        current_user.artists_like = str_list_to_list(artists_like)
        return {'status': 200}

    @app.route('/tracks_search', methods=['GET'])
    def tracks_search():
        query = request.args.get('query')
        return model_py.answer_song_search_query(query)
        
    @app.route('/get_recommendations', methods=['GET'])
    def get_recommendations():
        tracks_like = str_list_to_list(request.args.get('tracks'))
        users_prefs = mongo.db.users_preferencesDB
        existing_pref = users_prefs.find_one({'_id': current_user.id})
        genres_like = str_list_to_list(existing_pref['genres_like'])
        artists_like = str_list_to_list(existing_pref['artists_like'])
        result = get_recommends(artists_like,genres_like,tracks_like)
        return result

    @app.route('/register', methods=['POST', 'GET'])
    def register():
        if request.method == 'POST':
            login = request.form['login'].strip()
            password = request.form['pass'].strip() 

            users = mongo.db.usersDB
            
            existing_user = users.find_one(
                {'login': login}, {'_id': 0})
            if existing_user is None:
                logout_user()
                hashpass = bc.generate_password_hash(password).decode('utf-8')
                new_user = User(login)
                user_data_to_save = new_user.dict()
                user_data_to_save['password'] = hashpass
                if users.insert_one(user_data_to_save):
                    login_user(new_user)
                    return redirect(url_for('index'))
                else:
                    return redirect(url_for('register', error=2))

            return redirect(url_for('register', error=1))

        return render_template('register.html', error=request.args.get("error"))

    @app.route('/logout', methods=['GET'])
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('login'))

    # LOGIN MANAGER REQUIREMENTS

    # Load user from user ID
    @login_manager.user_loader
    def load_user(user_id):
        # Return user object or none
        users = mongo.db.usersDB
        user = users.find_one({'id': user_id}, {'_id': 0})
        if user:
            return User.make_from_dict(user)
        return None

    # Safe URL
    def is_safe_url(target):
        ref_url = urlparse(request.host_url)
        test_url = urlparse(urljoin(request.host_url, target))
        return test_url.scheme in ('http', 'https') and \
            ref_url.netloc == test_url.netloc


    # Heroku environment
    if os.environ.get('APP_LOCATION') == 'heroku':
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
    else:
        app.run(host='localhost', port=8080, debug=True)