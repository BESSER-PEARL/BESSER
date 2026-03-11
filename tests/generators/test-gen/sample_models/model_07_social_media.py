"""
BUML Model Example 7: Social Media Platform
A simple social media system with users, posts, and comments
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
public_lit = EnumerationLiteral(name="PUBLIC")
private_lit = EnumerationLiteral(name="PRIVATE")
friends_only_lit = EnumerationLiteral(name="FRIENDS_ONLY")

privacy_enum = Enumeration(
    name="PrivacyLevel",
    literals={public_lit, private_lit, friends_only_lit}
)

# =============================================================================
# 2. Define User Attributes
# =============================================================================
user_id_prop = Property(name="userId", type=StringType, multiplicity=Multiplicity(1, 1))
username_prop = Property(name="username", type=StringType, multiplicity=Multiplicity(1, 1))
bio_prop = Property(name="bio", type=StringType, multiplicity=Multiplicity(1, 1))
followers_count_prop = Property(name="followersCount", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Post Attributes
# =============================================================================
post_id_prop = Property(name="postId", type=StringType, multiplicity=Multiplicity(1, 1))
content_prop = Property(name="content", type=StringType, multiplicity=Multiplicity(1, 1))
timestamp_prop = Property(name="timestamp", type=StringType, multiplicity=Multiplicity(1, 1))
likes_count_prop = Property(name="likesCount", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Comment Attributes
# =============================================================================
comment_id_prop = Property(name="commentId", type=StringType, multiplicity=Multiplicity(1, 1))
text_prop = Property(name="text", type=StringType, multiplicity=Multiplicity(1, 1))
comment_timestamp_prop = Property(name="timestamp", type=StringType, multiplicity=Multiplicity(1, 1))
upvotes_prop = Property(name="upvotes", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define User Methods
# =============================================================================
create_profile_method = Method(
    name="createProfile",
    parameters=[],
    code="""
def createProfile(self):
    self.followersCount = 0
    print(f"Profile created for @{self.username}")
    print(f"Bio: {self.bio}")
"""
)

follow_user_method = Method(
    name="followUser",
    parameters=[Parameter(name="targetUser", type=StringType)],
    code="""
def followUser(self, targetUser):
    print(f"@{self.username} is now following @{targetUser}")
"""
)

update_bio_method = Method(
    name="updateBio",
    parameters=[Parameter(name="newBio", type=StringType)],
    code="""
def updateBio(self, newBio):
    self.bio = newBio
    print(f"Bio updated for @{self.username}")
"""
)

# =============================================================================
# 6. Define Post Methods
# =============================================================================
publish_method = Method(
    name="publish",
    parameters=[],
    code="""
def publish(self):
    self.likesCount = 0
    print(f"Post {self.postId} published at {self.timestamp}")
    print(f"Content: {self.content}")
"""
)

like_post_method = Method(
    name="likePost",
    parameters=[],
    code="""
def likePost(self):
    self.likesCount += 1
    print(f"Post {self.postId} liked. Total likes: {self.likesCount}")
"""
)

edit_content_method = Method(
    name="editContent",
    parameters=[Parameter(name="newContent", type=StringType)],
    code="""
def editContent(self, newContent):
    self.content = newContent
    print(f"Post {self.postId} has been edited")
"""
)

# =============================================================================
# 7. Define Comment Methods
# =============================================================================
post_comment_method = Method(
    name="postComment",
    parameters=[],
    code="""
def postComment(self):
    self.upvotes = 0
    print(f"Comment {self.commentId} posted at {self.timestamp}")
    print(f"Text: {self.text}")
"""
)

upvote_method = Method(
    name="upvote",
    parameters=[],
    code="""
def upvote(self):
    self.upvotes += 1
    print(f"Comment {self.commentId} upvoted. Total upvotes: {self.upvotes}")
"""
)

edit_comment_method = Method(
    name="editComment",
    parameters=[Parameter(name="newText", type=StringType)],
    code="""
def editComment(self, newText):
    self.text = newText
    print(f"Comment {self.commentId} has been edited")
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
user_class = Class(
    name="User",
    attributes={user_id_prop, username_prop, bio_prop, followers_count_prop},
    methods={create_profile_method, follow_user_method, update_bio_method}
)

post_class = Class(
    name="Post",
    attributes={post_id_prop, content_prop, timestamp_prop, likes_count_prop},
    methods={publish_method, like_post_method, edit_content_method}
)

comment_class = Class(
    name="Comment",
    attributes={comment_id_prop, text_prop, comment_timestamp_prop, upvotes_prop},
    methods={post_comment_method, upvote_method, edit_comment_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# User --< Post (one user can create many posts)
user_post_end = Property(
    name="author",
    type=user_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
user_posts_end = Property(
    name="posts",
    type=post_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
user_post_assoc = BinaryAssociation(
    name="Creates",
    ends={user_post_end, user_posts_end}
)

# Post --< Comment (one post can have many comments)
post_end = Property(
    name="post",
    type=post_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
post_comments_end = Property(
    name="comments",
    type=comment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
post_comment_assoc = BinaryAssociation(
    name="HasComments",
    ends={post_end, post_comments_end}
)

# User --< Comment (one user can write many comments)
user_comment_end = Property(
    name="commenter",
    type=user_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
user_comments_end = Property(
    name="myComments",
    type=comment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
user_comment_assoc = BinaryAssociation(
    name="Writes",
    ends={user_comment_end, user_comments_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
social_media_model = DomainModel(
    name="SocialMediaPlatform",
    types={user_class, post_class, comment_class, privacy_enum},
    associations={user_post_assoc, post_comment_assoc, user_comment_assoc}
)

print("✓ Social Media Platform BUML Model created successfully!")
print(f"  Classes: {[c.name for c in social_media_model.get_classes()]}")
print(f"  Associations: {[a.name for a in social_media_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=social_media_model, output_dir="output_social_media")
python_gen.generate()