# Mutant Operator: Removed Assignment
from datetime import datetime, date, time
from enum import Enum

class PrivacyLevel(Enum):
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'
    FRIENDS_ONLY = 'FRIENDS_ONLY'

class Comment:

    def __init__(self, commentId: str, text: str, timestamp: str, upvotes: int, post: 'Post'=None, commenter: 'User'=None):
        self.commentId = commentId
        self.text = text
        self.timestamp = timestamp
        self.upvotes = upvotes
        self.post = post
        self.commenter = commenter

    @property
    def upvotes(self) -> int:
        return self.__upvotes

    @upvotes.setter
    def upvotes(self, upvotes: int):
        self.__upvotes = upvotes

    @property
    def text(self) -> str:
        return self.__text

    @text.setter
    def text(self, text: str):
        self.__text = text

    @property
    def timestamp(self) -> str:
        return self.__timestamp

    @timestamp.setter
    def timestamp(self, timestamp: str):
        self.__timestamp = timestamp

    @property
    def commentId(self) -> str:
        return self.__commentId

    @commentId.setter
    def commentId(self, commentId: str):
        self.__commentId = commentId

    @property
    def post(self):
        return self.__post

    @post.setter
    def post(self, value):
        old_value = getattr(self, f'_Comment__post', None)
        pass
        if old_value is not None:
            if hasattr(old_value, 'comments'):
                opp_val = getattr(old_value, 'comments', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'comments'):
                opp_val = getattr(value, 'comments', None)
                if opp_val is None:
                    setattr(value, 'comments', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def commenter(self):
        return self.__commenter

    @commenter.setter
    def commenter(self, value):
        old_value = getattr(self, f'_Comment__commenter', None)
        self.__commenter = value
        if old_value is not None:
            if hasattr(old_value, 'myComments'):
                opp_val = getattr(old_value, 'myComments', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'myComments'):
                opp_val = getattr(value, 'myComments', None)
                if opp_val is None:
                    setattr(value, 'myComments', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def upvote(self):
        self.upvotes += 1
        print(f'Comment {self.commentId} upvoted. Total upvotes: {self.upvotes}')

    def postComment(self):
        self.upvotes = 0
        print(f'Comment {self.commentId} posted at {self.timestamp}')
        print(f'Text: {self.text}')

    def editComment(self, newText):
        self.text = newText
        print(f'Comment {self.commentId} has been edited')

class Post:

    def __init__(self, postId: str, content: str, timestamp: str, likesCount: int, author: 'User'=None, comments: set['Comment']=None):
        self.postId = postId
        self.content = content
        self.timestamp = timestamp
        self.likesCount = likesCount
        self.author = author
        self.comments = comments if comments is not None else set()

    @property
    def content(self) -> str:
        return self.__content

    @content.setter
    def content(self, content: str):
        self.__content = content

    @property
    def postId(self) -> str:
        return self.__postId

    @postId.setter
    def postId(self, postId: str):
        self.__postId = postId

    @property
    def likesCount(self) -> int:
        return self.__likesCount

    @likesCount.setter
    def likesCount(self, likesCount: int):
        self.__likesCount = likesCount

    @property
    def timestamp(self) -> str:
        return self.__timestamp

    @timestamp.setter
    def timestamp(self, timestamp: str):
        self.__timestamp = timestamp

    @property
    def author(self):
        return self.__author

    @author.setter
    def author(self, value):
        old_value = getattr(self, f'_Post__author', None)
        self.__author = value
        if old_value is not None:
            if hasattr(old_value, 'posts'):
                opp_val = getattr(old_value, 'posts', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'posts'):
                opp_val = getattr(value, 'posts', None)
                if opp_val is None:
                    setattr(value, 'posts', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def comments(self):
        return self.__comments

    @comments.setter
    def comments(self, value):
        old_value = getattr(self, f'_Post__comments', None)
        self.__comments = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'post'):
                    opp_val = getattr(item, 'post', None)
                    if opp_val == self:
                        setattr(item, 'post', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'post'):
                    opp_val = getattr(item, 'post', None)
                    setattr(item, 'post', self)

    def likePost(self):
        self.likesCount += 1
        print(f'Post {self.postId} liked. Total likes: {self.likesCount}')

    def publish(self):
        self.likesCount = 0
        print(f'Post {self.postId} published at {self.timestamp}')
        print(f'Content: {self.content}')

    def editContent(self, newContent):
        self.content = newContent
        print(f'Post {self.postId} has been edited')

class User:

    def __init__(self, bio: str, followersCount: int, userId: str, username: str, posts: set['Post']=None, myComments: set['Comment']=None):
        self.bio = bio
        self.followersCount = followersCount
        self.userId = userId
        self.username = username
        self.posts = posts if posts is not None else set()
        self.myComments = myComments if myComments is not None else set()

    @property
    def followersCount(self) -> int:
        return self.__followersCount

    @followersCount.setter
    def followersCount(self, followersCount: int):
        self.__followersCount = followersCount

    @property
    def userId(self) -> str:
        return self.__userId

    @userId.setter
    def userId(self, userId: str):
        self.__userId = userId

    @property
    def bio(self) -> str:
        return self.__bio

    @bio.setter
    def bio(self, bio: str):
        self.__bio = bio

    @property
    def username(self) -> str:
        return self.__username

    @username.setter
    def username(self, username: str):
        self.__username = username

    @property
    def posts(self):
        return self.__posts

    @posts.setter
    def posts(self, value):
        old_value = getattr(self, f'_User__posts', None)
        self.__posts = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'author'):
                    opp_val = getattr(item, 'author', None)
                    if opp_val == self:
                        setattr(item, 'author', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'author'):
                    opp_val = getattr(item, 'author', None)
                    setattr(item, 'author', self)

    @property
    def myComments(self):
        return self.__myComments

    @myComments.setter
    def myComments(self, value):
        old_value = getattr(self, f'_User__myComments', None)
        self.__myComments = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'commenter'):
                    opp_val = getattr(item, 'commenter', None)
                    if opp_val == self:
                        setattr(item, 'commenter', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'commenter'):
                    opp_val = getattr(item, 'commenter', None)
                    setattr(item, 'commenter', self)

    def updateBio(self, newBio):
        self.bio = newBio
        print(f'Bio updated for @{self.username}')

    def createProfile(self):
        self.followersCount = 0
        print(f'Profile created for @{self.username}')
        print(f'Bio: {self.bio}')

    def followUser(self, targetUser):
        print(f'@{self.username} is now following @{targetUser}')