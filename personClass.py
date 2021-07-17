class person:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

    def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

    def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

def update(self, rects):
	# check to see if the list of input bounding box rectangles
	# is empty
	if len(rects) == 0:
		# loop over any existing tracked objects and mark them
		# as disappeared
		for objectID in list(self.disappeared.keys()):
			self.disappeared[objectID] += 1
			# if we have reached a maximum number of consecutive
			# frames where a given object has been marked as
			# missing, deregister it
			if self.disappeared[objectID] > self.maxDisappeared:
				self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
    	return self.objects

 