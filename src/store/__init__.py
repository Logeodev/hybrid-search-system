import redis, struct

def to_binary(vector):
    return struct.pack('>' + 'f'*len(vector), *vector)

redis_client = redis.Redis(host='localhost', port=6379, db=0)