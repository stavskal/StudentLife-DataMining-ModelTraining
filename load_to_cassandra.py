from cassandra.cluster import Cluster

cluster = Cluster()

session = cluster.connect('demo')

session.execute("CREATE TABLE mamamia (time_stamp int,act int, PRIMARY KEY(time_stamp));")

# TODO: replicate all my postgresql inserts > cassandra cql
# IMPORTANT: primary key needs to be specified, not assigned automatically