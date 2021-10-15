import numpy as np
import pandas as pd
from multiprocessing import Pool, Process
#import cudf

from scipy.spatial.distance import cosine

print(__name__) 

def write_cos_dist(t):
    print(t[0].shape)
    print(t[1].shape)
    res = t[0].apply(cosine, axis=1, args=(t[1],))
    return res

if __name__ == 'predict':

    from collections import Counter
    import math
    import re
    import model as model_py
    import threading
    import multiprocessing

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
            #print(bests_all_time_df)
            songs_to_boost = pd.concat([bests_from_popular_artists_df_sorted.sort_values('likes_count', ascending=False)[:350], 
                                        bests_all_time_df]).sample(25, random_state=100)

            #print(songs_to_boost)
            model_track_weights = {track_id:0 for track_id in self.df_song_names.index}

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
            #for index,row in self.df_users_preferences.iterrows():
            #    cosine_dists[index] = cosine(user, row)
            
            cosine_dists_sorted = sorted(list(cosine_dists.items()), key=lambda x:x[1])

            users_neighbors_to_consider = [(user_dist[0], 1-user_dist[1])
                                        for user_dist in cosine_dists_sorted[:self.neighbors_users_count]]
            users_neighbors = [t[0] for t in users_neighbors_to_consider]

            self.songs_neighbors = []
            add_neighbor_tracks_v = np.vectorize(self.add_neighbor_tracks)
            add_neighbor_tracks_v(users_neighbors)
            song_id_vs_neighbor_repeats = sorted(list(dict(Counter(self.songs_neighbors)).items()), key=lambda x:1/x[1])

            relevant_artists = self.df_song_names.loc[self.songs_neighbors]['artist'].value_counts(normalize=True).head(50)

            relevant_artists_normalized = relevant_artists/(relevant_artists[0]/self.artists_importance)

            relevant_genres = self.df_song_names.loc[self.songs_neighbors]['genre'].value_counts(normalize=True).head(5)
            relevant_genres = relevant_genres/(relevant_genres[0]/self.genres_importance)
            relevant_genres.append(pd.Series({genre_chosen:self.genres_importance for genre_chosen in self.genres_chosen}))
            
            all_sim_songs = []

            for track_id_lst in self.df_song_names.loc[[track_id for track_id in list_user_tracks_id
                                                    if track_id in self.df_songs.index]]['bests_from_album']:
                no_brackets = track_id_lst.replace(']', '').replace('[', '').replace(' ', '')
                if len(no_brackets) > 0:
                    all_sim_songs.extend([int(t) for t in no_brackets.split(',')])

            all_sim_songs = set(all_sim_songs)
            
            self.songs_model_weights_user_based = {sim_song:self.sim_songs_importance for sim_song in all_sim_songs}
            self.genres_model_weights = relevant_genres
            self.artists_model_weights = relevant_artists_normalized
        
        def artists_vs_weight_into_songs_vs_weights(self, dict_artists_vs_weights):
            result = {}
            for artist, weight in dict_artists_vs_weights.items():
                for track_id in self.df_song_names[self.df_song_names['artist'] == artist].index:
                    result[track_id] = weight
            return result
        
        def genres_vs_weight_into_songs_vs_weights(self, dict_genres_vs_weights):
            result = {}
            for genre, weight in dict_genres_vs_weights.items():
                for track_id in self.df_song_names[self.df_song_names['genre'] == genre].index:
                    result[track_id] = weight
            return result
        
        #def compute_
        
        def apply_song_weights(self, dict_song_ids_vs_weights, coeff):
            #song_ids_vs_weights = pd.Series(dict_song_ids_vs_weights)
            #result_weights.apply()
            for track_id, weight in dict_song_ids_vs_weights.items():
                self.result_weights[track_id] = 1. - (1. - self.result_weights[track_id])*(1.-1.*weight*coeff)
            pass
        
        def fit(self, genres_chosen, artists_chosen, tracks_chosen, 
                learning_rate=0.05, 
                artists_importance=0.15, 
                genres_importance=0.25,
                sim_songs_importance=0.10,
                sim_artists_importance=0.15):
            
            self.artists_importance = artists_importance
            self.genres_importance = genres_importance
            self.sim_songs_importance = sim_songs_importance
            self.sim_artists_importance = sim_artists_importance
            self.tracks_chosen = tracks_chosen
            self.genres_chosen = tracks_chosen
            if len(tracks_chosen) > 0:
                print(tracks_chosen)
                self.user_based_model(tracks_chosen)
                    
            coldstart_coeff = self.coldstart_coeff_sigmoid(len(tracks_chosen))
            
            self.result_weights = {track_id:0 for track_id in self.df_song_names.index}  
            
            all_sim_artists = []
            for artist_name in artists_chosen:
                similar_artists = [x[0] for x in sorted(
                    list(
                        dict(
                            Counter(
                                df_users[
                                    df_users['owner_id'].isin(df_users[df_users['artist_name'] == artist_name]['owner_id'].unique())]
                                ['artist_name'])).items()), key=lambda x: 1/x[1])]
                try:
                    similar_artists.remove(artist_name)
                except Exception:
                    pass
                similar_artists = similar_artists[:5]
                all_sim_artists.extend(similar_artists)
                #print(artist_name)

            #all_sim_artists.extend(artists_chosen)
            self.artists_model_weights_user_based = {artist_name:self.sim_artists_importance for artist_name in all_sim_artists}
            #print(self.df_song_names.loc[self.songs_model_weights_user_based.keys()])
            #print(self.genres_model_weights)
            #print(self.artists_model_weights)
            #print('begin')
            self.apply_song_weights(self.songs_model_weights_coldstart, coldstart_coeff)
            self.apply_song_weights(self.songs_model_weights_user_based, 1. - coldstart_coeff)
            self.apply_song_weights(self.genres_vs_weight_into_songs_vs_weights(self.genres_model_weights), 
                                    1. - coldstart_coeff)
            self.apply_song_weights(self.artists_vs_weight_into_songs_vs_weights(self.artists_model_weights), 
                                    1. - coldstart_coeff)
            #print(self.artists_model_weights_user_based)
            self.apply_song_weights(self.artists_vs_weight_into_songs_vs_weights(self.artists_model_weights_user_based), 
                                    1. - coldstart_coeff)
            pass
        
        def update_dists(self, track_id_vs_coeff):
            try:
                self.all_cos_distances[track_id_vs_coeff[0]] = self.all_cos_distances[track_id_vs_coeff[0]]*(1.-track_id_vs_coeff[1])
            except Exception:
                pass

        def predict(self, predict_count=20):
            print('cos_dist_start')
            
            global user

            #num_workers = 8
            #pool = Pool(num_workers, initargs={'name': 'helper'})
            #len_df = recommender.df_songs.shape[0]
            #self.all_cos_distances = pd.concat(pool.map(write_cos_dist, [(recommender.df_songs[int(len_df*i/num_workers):int(len_df*(i+1)/num_workers)], user) for i in range(num_workers)]))
            self.all_cos_distances = write_cos_dist((self.df_songs, user))
            #pool.close()
            #pool.join()

            print('cos_dist_end')
            
            #print('a')
            update_dists_v = np.vectorize(self.update_dists)
            update_dists_v(self.result_weights.items())
            #for track_id, coeff in self.result_weights.items():
            #    try:
            #        self.all_cos_distances[track_id] = self.all_cos_distances[track_id]*(1.-coeff)
            #    except Exception:
            #        pass
            #print('b')
            track_ids_sorted = [t[0] for t in sorted(list(self.all_cos_distances.items()), key=lambda x: x[1])]
            
            for track_id_already_exist in self.tracks_chosen:
                try:
                    track_ids_sorted.remove(track_id_already_exist)
                except Exception:
                    pass
            return self.df_song_names.loc[track_ids_sorted][:predict_count*5].sample(predict_count)  
    

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
