import pandas as pd
import numpy as np

df_songs = pd.read_csv('../data/songs_dataset_transformed.csv', ',', index_col='track_id')
df_song_names = pd.read_csv('../data/songs_dataset.csv',';', encoding='cp1251', index_col='track_id').drop_duplicates()
df_users = pd.read_csv('../data/users_dataset.csv', ';', encoding='cp1251')
df_users_preferences = pd.read_csv('../data/users_preferences_dataset.csv', index_col='user_id')
df_song_names_user_valid = df_song_names.loc[[i for i in df_users['track_id'] if i in df_song_names.index]]
df_names = pd.read_csv('../data/song_names_dataset.csv', index_col='track_id')

df_names['artist_title'] = df_names['artist'] + ' ' + df_names['title']
df_names['artist_title'] = [str(x).lower() for x in df_names['artist_title']]

genres = {'genres':[{'genre':'pop','picture': 'sweet.png','rus': 'Поп'}, 
             {'genre':'foreignrap','picture': 'rap.png','rus': 'Зарубежный рэп'},
             {'genre':'electronics','picture': 'loud.png', 'rus':'Электроника'},
             {'genre':'rock','picture': 'rock.png','rus':'Рок'},
             {'genre':'dance','picture': 'energy.png','rus':'Танцевальная'},
             {'genre':'classical','picture': 'estrada.png','rus':'Классика'},
             {'genre':'jazz','picture': 'vocals.png','rus':'Джаз'},
             {'genre':'alternative','picture': 'exclusive.png','rus':'Альтернатива'},
             {'genre':'rusrap','picture': 'rap.png','rus':'Русский рэп'},
             {'genre':'house','picture': 'loud.png','rus':'Хаус'},
             {'genre':'folk','picture': 'estrada.png','rus':'Народные'},
             {'genre':'indie','picture': 'indie.png','rus':'Инди'},
             {'genre':'newage','picture': 'exclusive.png','rus':'Newage'},
             {'genre':'newwave', 'picture':'exclusive.png','rus':'Newwave'},
             {'genre':'ruspop','picture': 'sweet.png','rus':'Русский поп'},
             {'genre':'rnb','picture': 'electroguitar.png','rus':'R\'n\'B'},
             {'genre':'rusrock','picture': 'rock.png','rus':'Русский рок'},
             {'genre':'rusestrada','picture': 'estrada.png','rus':'Русская эстрада'},
             {'genre':'relax','picture': 'cool.png','rus':'Релакс'},
             {'genre':'country','picture': 'electroguitar.png','rus':'Кантри'},
             {'genre':'trance','picture': 'cool.png','rus':'Транс'},
             {'genre': 'punk','picture': 'punk.png','rus':'Панк'},
             {'genre':'reggae','picture': 'indie.png','rus':'Регги'},
             {'genre':'extrememetal','picture': 'electroguitar.png','rus':'Экстрим-метал'},
             {'genre':'metal','picture': 'electroguitar.png','rus':'Метал'},
             {'genre':'blues','picture': 'electroguitar.png','rus':'Блюз'},
             {'genre':'techno','picture': 'cool.png','rus':'Техно'},
             {'genre':'hardrock','picture': 'electroguitar.png','rus':'Хард-рок'},
             {'genre':'shanson','picture': 'estrada.png','rus':'Шансон'},
             {'genre':'estrada','picture': 'estrada.png','rus':'Эстрада'},
             {'genre':'local-indie','picture': 'indie.png','rus':'Локал-инди'},
             {'genre':'classicmetal','picture': 'electroguitar.png','rus':'Классика метала'},
             {'genre':'lounge','picture': 'cool.png','rus':'Лаундж'},
             {'genre':'rnr','picture': 'electroguitar.png','rus':'Рок-н-ролл'},
             {'genre':'posthardcore','picture': 'electroguitar.png','rus':'Пост-хардкор'},
             {'genre':'folkrock','picture': 'rock.png','rus':'Фолк-рок'},
             {'genre':'rap','picture': 'rap.png','rus':'Рэп'},
             {'genre':'industrial','picture': 'electroguitar.png','rus':'Индастриал'},
             {'genre':'numetal','picture': 'electroguitar.png','rus':'Ню-метал'},
             {'genre':'hardcore','picture': 'electroguitar.png','rus':'Хардкор'},
             {'genre':'progmetal','picture': 'electroguitar.png','rus':'Прогрессив-метал'},
             {'genre':'funk','picture': 'loud.png','rus':'Фанк'},
             {'genre':'reggaeton','picture': 'vocals.png','rus':'Реггетон'},
             {'genre':'ukrrock','picture': 'rock.png','rus':'Украинский рок'},
             {'genre':'rusfolk','picture': 'vocals.png','rus':'Русские народные'},
             {'genre':'rusbards','picture': 'indie.png','rus':'Русские барды'},
             {'genre':'ska','picture': 'electroguitar.png','rus':'Ска'},
             {'genre':'folkmetal','picture': 'electroguitar.png','rus':'Фолк-метал'},
             {'genre':'fiction','picture': 'cool.png','rus':'Фикшн'},
             {'genre':'allrock','picture': 'electroguitar.png','rus':'Allrock'},
             {'genre':'disco','picture': 'energy.png','rus':'Диско'},
             ]
}

def return_artists(genres_lst_str):
    genres_lst = genres_lst_str.replace('[', '').replace(']', '').replace('\"', '').split(',')
    result_lst = []
    for genre in genres_lst:
        result_lst.extend([nameid_count[0] for nameid_count in [(t[0],t[1])
                        for t in sorted(list(dict(df_song_names_user_valid[
                            (df_song_names_user_valid['genre'] == genre) &
                            (df_song_names_user_valid['artist'] != 'сборник')]
                                        [['artist', 'artist_id']].value_counts()[:50]).items()),
                                        key=lambda x: 1/x[1])]])
    result_obj = {'artists': [{'artist': t[0], 'picture': str(t[1])+'.jpg', 'artist_id': t[1]} for t in result_lst]}
    return result_obj

def has_substr(s, q):
    return q in str(s)

def answer_song_search_query(query_str):
    return {'results': [{'track_id': ind,
                         'artist': info.iloc[0],
                         'title':info.iloc[1],
                         'artist_id':info.iloc[2],
                         } for ind,info
                         in df_names[
                             df_names['artist_title'].apply(has_substr,
                                                            args=(query_str,))
                                    ][:10][['artist','title', 'artist_id']].iterrows()
                        ]
            }
