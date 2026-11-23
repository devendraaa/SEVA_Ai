from indexing import load_vectorstores

discovery_db, detail_db = load_vectorstores()

print("Discovery:", discovery_db._collection.count())
print("Detail:", detail_db._collection.count())

assert discovery_db._collection.count() == 23438
assert detail_db._collection.count() == 67945

print("Existing vector stores verified successfully!")