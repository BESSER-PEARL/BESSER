from abc import ABC
from enum import Enum
import json
from typing import Any, Callable, Optional, Union

from besser.BUML.metamodel.state_machine.state_machine import Action, Transition, Event, Condition, StateMachine, State, Session, TransitionBuilder
from besser.BUML.metamodel.structural import NamedElement


class File:
    """A representation of files sent and received by a agent.

    Files are used to encapsulate information about the files exchanged in a agent conversation. They include
    attributes such as the file's name, type, and base64 representation.
    Note that at least one of path, data or base64 need to be set.

    Args:
        file_name (str): The name of the file.
        file_type (str): The type of the file.
        file_base64 (str, optional): The base64 representation of the file.
        file_path (str, optional): Path to the file.
        file_data (bytes, optional): Raw file data.

    Attributes:
        name (str): The name of the file.
        type (str): The type of the file.
        base64 (str): The base64 representation of the file.
    """

    def __init__(self, file_name: str = None, file_type: str = None, file_base64: str = None, file_path: str = None, file_data: bytes = None):
        self.name: str = file_name
        self.type: str = file_type
        self.base64: str = file_base64


class AgentReply(Action):
    """Primitive action that represents sending a reply message.

    Args:
        message (Expression): The message to send (can be Literal, VariableRef, ParameterRef, or any Expression)

    Attributes:
        message (Expression): The message expression
    """

    def __init__(self, message: str):
        self.message: str = message

    def __repr__(self):
        return f"AgentReply(message={self.message!r})"


class LLMReply(Action):
    """Primitive action that represents sending a reply using an LLM.

    Args:
        prompt (str, optional): Additional system prompt injected when calling the LLM.

    Attributes:
        prompt (str | None): Optional system prompt that augments the user message.
    """

    def __init__(self, prompt: Optional[str] = None):
        super().__init__()
        self.prompt: Optional[str] = prompt

    def __repr__(self):
        return f"LLMReply(prompt={self.prompt!r})"


class RAGReply(Action):
    """Primitive action that represents sending a reply using a configured RAG pipeline.

    Args:
        rag_db_name (str): The logical name of the RAG database to query.
        prompt (str, optional): Optional instructions passed to the LLM phase of the RAG answer.

    Attributes:
        rag_db_name (str): Identifier of the RAG database that should handle the reply.
        prompt (str | None): Additional instructions for the downstream LLM.
    """

    def __init__(self, rag_db_name: str, prompt: Optional[str] = None):
        super().__init__()
        self.rag_db_name: str = rag_db_name
        self.prompt: Optional[str] = prompt

    def __repr__(self):
        return f"RAGReply(rag_db_name={self.rag_db_name!r}, prompt={self.prompt!r})"


class IntentClassifierConfiguration(ABC):
    """The Intent Classifier Configuration abstract class.

    This configuration is assigned to a state, allowing the customization of its intent classifier.

    This class serves as a template to implement intent classifier configurations for the different Intent Classifiers.
    """

    def __init__(self):
        pass


class SimpleIntentClassifierConfiguration(IntentClassifierConfiguration):
    """The Simple Intent Classifier Configuration class.

    Args:
        num_words (int): Max num of words to keep in the index of words
        num_epochs (int): Number of epochs to be run during training
        embedding_dim (int): Number of embedding dimensions to be used when embedding the words
        input_max_num_tokens (int): Max length for the vector representing a sentence
        discard_oov_sentences (bool): whether to automatically assign zero probabilities to sentences with all tokens
            being oov ones or not
        check_exact_prediction_match (bool): Whether to check for exact match between the sentence to predict and one of
            the training sentences or not
        activation_last_layer (str): The activation function of the last layer
        activation_hidden_layers (str): The activation function of the hidden layers
        lr (float): Learning rate for the optimizer

    Attributes:
        num_words (int): Max num of words to keep in the index of words
        num_epochs (int): Number of epochs to be run during training
        embedding_dim (int): Number of embedding dimensions to be used when embedding the words
        input_max_num_tokens (int): Max length for the vector representing a sentence
        discard_oov_sentences (bool): whether to automatically assign zero probabilities to sentences with all tokens
            being oov ones or not
        check_exact_prediction_match (bool): Whether to check for exact match between the sentence to predict and one of
            the training sentences or not
        activation_last_layer (str): The activation function of the last layer
        activation_hidden_layers (str): The activation function of the hidden layers
        lr (float): Learning rate for the optimizer
    """

    def __init__(
            self,
            num_words: int = 1000,
            num_epochs: int = 300,
            embedding_dim: int = 128,
            input_max_num_tokens: int = 15,
            discard_oov_sentences: bool = True,
            check_exact_prediction_match: bool = True,
            activation_last_layer: str = 'sigmoid',
            activation_hidden_layers: str = 'tanh',
            lr: float = 0.001,
    ):
        super().__init__()
        self.name = 'simple'
        self.num_words: int = num_words
        self.num_epochs: int = num_epochs
        self.embedding_dim: int = embedding_dim
        self.input_max_num_tokens: int = input_max_num_tokens
        self.discard_oov_sentences: bool = discard_oov_sentences
        self.check_exact_prediction_match: bool = check_exact_prediction_match
        self.activation_last_layer: str = activation_last_layer
        self.activation_hidden_layers: str = activation_hidden_layers
        self.lr: float = lr


class LLMSuite(Enum):
    """Enumeration of the available LLM suites."""

    openai = "openai"
    huggingface = "huggingface"
    huggingface_inference_api = "huggingface-inference-api"
    replicate = "replicate"


class LLMIntentClassifierConfiguration(IntentClassifierConfiguration):
    """The LLM Intent Classifier Configuration class.

    Args:
        llm_suite (LLMSuite): the service provider from which we will load/access the LLM
        parameters (dict): the LLM parameters (this will vary depending on the suite and the LLM)
        use_intent_descriptions (bool): whether to include the intent descriptions in the LLM prompt
        use_training_sentences (bool): whether to include the intent training sentences in the LLM prompt
        use_entity_descriptions (bool): whether to include the entity descriptions in the LLM prompt
        use_entity_synonyms (bool): whether to include the entity value's synonyms in the LLM prompt

    Attributes:
        llm_suite (str): the service provider from which we will load/access the LLM
        parameters (dict): the LLM parameters (this will vary depending on the suite and the LLM)
        use_intent_descriptions (bool): whether to include the intent descriptions in the LLM prompt
        use_training_sentences (bool): whether to include the intent training sentences in the LLM prompt
        use_entity_descriptions (bool): whether to include the entity descriptions in the LLM prompt
        use_entity_synonyms (bool): whether to include the entity value's synonyms in the LLM prompt
    """

    def __init__(
            self,
            llm_suite: LLMSuite,
            parameters: dict = {},
            use_intent_descriptions: bool = False,
            use_training_sentences: bool = False,
            use_entity_descriptions: bool = False,
            use_entity_synonyms: bool = False
    ):
        super().__init__()
        self.llm_suite: str = llm_suite.value
        self.parameters: dict = parameters
        self.use_intent_descriptions: bool = use_intent_descriptions
        self.use_training_sentences: bool = use_training_sentences
        self.use_entity_descriptions: bool = use_entity_descriptions
        self.use_entity_synonyms: bool = use_entity_synonyms


class LLMWrapper(ABC):
    """The LLM Wrapper class.

    Args:
        name (str): the LLM name
        parameters (dict): the LLM parameters (this will vary depending on the LLM)
        global_context (str): the global context to be used in the LLM prompt

    Attributes:
        name (str): the LLM name
        parameters (dict): the LLM parameters (this will vary depending on the LLM)
        global_context (str): the global context to be used in the LLM prompt
    """

    def __init__(
            self,
            name: str,
            agent: 'Agent',
            parameters: dict,
            global_context: str = None
    ):
        self.name: str = name
        self.parameters: dict = parameters
        self.global_context: str = global_context
        self._user_context: dict = dict()
        self._user_contexts: dict = dict()

        agent.llms.append(self)

    def predict(self, message: str, parameters: dict = None,
                system_message: str = None) -> str:
        """Make a prediction, i.e., generate an output.

        Args:
            message (Any): the LLM input text
            session (Session): the ongoing session, can be None if no context needs to be applied
            parameters (dict): the LLM parameters to use in the prediction. If none is provided, the default LLM
                parameters will be used
            system_message (str): system message to give high priority context to the LLM

        Returns:
            str: the LLM output
        """
        return None

    def chat(self, parameters: dict = None, system_message: str = None) -> str:
        """Make a prediction, i.e., generate an output.

        This function can provide the chat history to the LLM for the output generation, simulating a conversation or
        remembering previous messages.

        Args:
            session (Session): the user session
            parameters (dict): the LLM parameters. If none is provided, the RAG's default value will be used
            system_message (str): system message to give high priority context to the LLM
       
        Returns:
            str: the LLM output
        """
        return None

    def add_user_context(self, context: str, context_name: str) -> None:
        """Add user-specific context.

        Args:
            session (Session): the ongoing session
            context (str): the user-specific context
            context_name (str): the key given to the specific user context
        """

    def remove_user_context(self, context_name: str) -> None:
        """Remove user-specific context.

        Args:
            session (Session): the ongoing session
            context_name (str): the key given to the specific user context
        """


class LLMHuggingFace(LLMWrapper):
    """A HuggingFace LLM wrapper.

    Normally, we consider an LLM in HuggingFace those models under the tasks ``text-generation`` or
    ``text2text-generation`` tasks (`more info <https://huggingface.co/tasks/text-generation>`_), but there could be
    exceptions for other tasks (which have not been tested in this class).

    Args:
        agent (Agent): the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        global_context (str): the global context to be provided to the LLM for each request


    Attributes:
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        _global_context (str): the global context to be provided to the LLM for each request
    """

    def __init__(self, agent: 'Agent', name: str, parameters: dict, num_previous_messages: int = 1,
                 global_context: str = None):
        super().__init__(name, agent, parameters, global_context=global_context)
        self.agent: 'Agent' = agent
        self.num_previous_messages: int = num_previous_messages

    def set_model(self, name: str) -> None:
        """Set the LLM model name.

        Args:
            name (str): the new LLM name
        """
        self.name = name

    def set_num_previous_messages(self, num_previous_messages: int) -> None:
        """Set the number of previous messages to use in the chat functionality

        Args:
            num_previous_messages (int): the new number of previous messages
        """
        self.num_previous_messages = num_previous_messages


class LLMHuggingFaceAPI(LLMWrapper):
    """A HuggingFace LLM wrapper for HuggingFace's Inference API.

    Normally, we consider an LLM in HuggingFace those models under the tasks ``text-generation`` or
    ``text2text-generation`` tasks (`more info <https://huggingface.co/tasks/text-generation>`_), but there could be
    exceptions for other tasks (which have not been tested in this class).

    Args:
        agent (Agent): the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        global_context (str): the global context to be provided to the LLM for each request


    Attributes:
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        _global_context (str): the global context to be provided to the LLM for each request
    """

    def __init__(self, agent: 'Agent', name: str, parameters: dict, num_previous_messages: int = 1,
                 global_context: str = None):
        super().__init__(name, agent, parameters, global_context=global_context)
        self.agent: 'Agent' = agent
        self.num_previous_messages: int = num_previous_messages

    def set_model(self, name: str) -> None:
        """Set the LLM model name.

        Args:
            name (str): the new LLM name
        """
        self.name = name

    def set_num_previous_messages(self, num_previous_messages: int) -> None:
        """Set the number of previous messages to use in the chat functionality

        Args:
            num_previous_messages (int): the new number of previous messages
        """
        self.num_previous_messages = num_previous_messages


class LLMOpenAI(LLMWrapper):
    """An LLM wrapper for OpenAI's LLMs through its API.

    Args:
        agent (Agent): the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        global_context (str): the global context to be provided to the LLM for each request

    Attributes:
        _nlp_engine (NLPEngine): the NLPEngine that handles the NLP processes of the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0).
        _global_context (str): the global context to be provided to the LLM for each request
        _user_context (dict): user specific context to be provided to the LLM for each request
    """

    def __init__(self, agent: 'Agent', name: str, parameters: dict, num_previous_messages: int = 1,
                 global_context: str = None):
        super().__init__(name, agent, parameters, global_context=global_context)
        self.agent: 'Agent' = agent
        self.num_previous_messages: int = num_previous_messages

    def set_model(self, name: str) -> None:
        """Set the LLM model name.

        Args:
            name (str): the new LLM name
        """
        self.name = name

    def set_num_previous_messages(self, num_previous_messages: int) -> None:
        """Set the number of previous messages to use in the chat functionality

        Args:
            num_previous_messages (int): the new number of previous messages
        """
        self.num_previous_messages = num_previous_messages


class LLMReplicate(LLMWrapper):
    """An LLM wrapper for Replicate's LLMs through its API.

    Args:
        agent (Agent): the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0)
        global_context (str): the global context to be provided to the LLM for each request

    Attributes:
        _nlp_engine (NLPEngine): the NLPEngine that handles the NLP processes of the agent the LLM belongs to
        name (str): the LLM name
        parameters (dict): the LLM parameters
        num_previous_messages (int): for the chat functionality, the number of previous messages of the conversation
            to add to the prompt context (must be > 0)
        _global_context (str): the global context to be provided to the LLM for each request
        _user_context (dict): user specific context to be provided to the LLM for each request
    """

    def __init__(self, agent: 'Agent', name: str, parameters: dict, num_previous_messages: int = 1,
                 global_context: str = None):
        super().__init__(name, agent, parameters, global_context=global_context)
        self.agent: 'Agent' = agent
        self.num_previous_messages: int = num_previous_messages

    def set_model(self, name: str) -> None:
        """Set the LLM model name.

        Args:
            name (str): the new LLM name
        """
        self.name = name

    def set_num_previous_messages(self, num_previous_messages: int) -> None:
        """Set the number of previous messages to use in the chat functionality

        Args:
            num_previous_messages (int): the new number of previous messages
        """
        self.num_previous_messages = num_previous_messages


class RAGVectorStore:
    """Declarative description of a vector store used by a RAG pipeline.

    Args:
        embedding_provider (str): Identifier of the embedding backend (e.g., "openai", "huggingface").
        embedding_parameters (dict or None): Backend-specific parameters (API key placeholder, model name, etc.).
        persist_directory (str or None): Directory where the embedding index is stored.

    Attributes:
        embedding_provider (str): Provider used to build embeddings.
        embedding_parameters (dict): Provider specific parameters.
        persist_directory (str or None): Optional path used to persist vectors.
    """

    def __init__(self, embedding_provider: str, embedding_parameters: Optional[dict] = None, persist_directory: Optional[str] = None):
        self.embedding_provider: str = embedding_provider
        self.embedding_parameters: dict = embedding_parameters or {}
        self.persist_directory: Optional[str] = persist_directory


class RAGTextSplitter:
    """Declarative description of the chunking strategy used before indexing content."""

    def __init__(self, splitter_type: str, chunk_size: int, chunk_overlap: int):
        """Create a splitter definition.

        Args:
            splitter_type (str): The splitter implementation identifier (e.g., "recursive_character").
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Characters shared between adjacent chunks.
        """

        self.splitter_type: str = splitter_type
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap


class RAG(NamedElement):
    """Retrieval-Augmented Generation configuration bound to an agent.

    This models the minimal information required to generate code similar to::

        vector_store = Chroma(...)
        splitter = RecursiveCharacterTextSplitter(...)
        rag = RAG(agent=agent, vector_store=vector_store, splitter=splitter, llm_name='gpt-4o-mini', k=4, num_previous_messages=0)

    Args:
        name (str): Logical name of the RAG resource.
        agent (Agent): Agent that owns the configuration.
        vector_store (RAGVectorStore): Vector store definition.
        splitter (RAGTextSplitter): Chunking strategy definition.
        llm_name (str): Identifier of the LLM used to synthesize answers.
        k (int): Number of chunks retrieved per question.
        num_previous_messages (int): Conversation context depth forwarded to the LLM.
    """

    def __init__(
            self,
            name: str,
            agent: 'Agent',
            vector_store: RAGVectorStore,
            splitter: RAGTextSplitter,
            llm_name: str,
            k: int = 4,
            num_previous_messages: int = 0,
    ):
        super().__init__(name)
        self.agent: 'Agent' = agent
        self.vector_store: RAGVectorStore = vector_store
        self.splitter: RAGTextSplitter = splitter
        self.llm_name: str = llm_name
        self.k: int = k
        self.num_previous_messages: int = num_previous_messages


class AgentSession(Session):
    """A user session in a agent execution.

    When a user starts interacting with a state machine, a session is assigned to him/her to store user related
    information, such as the current state of the agent or any custom variable. A session can be accessed from the body of
    the states to read/write user information. If a state machine does not have the concept of 'users' (i.e., there are
    no concurrent executions of the state machine, but a single one) then it could simply have 1 unique session.

    Attributes:
        id (str): Inherited from Session, the session id, which must unique among all state machine sessions
        current_state (str): Inherited from Session, the current state in the state machine for this session
        message (str): The last message sent to the agent by this session
        predicted_intent (IntentClassifierPrediction): The last predicted intent for this session
        file (File): The last file sent to the agent.
        chat_history (list[tuple[str, int]]): The session chat history
    """

    def __init__(self):
        super().__init__()
        self.message: str = None
        self.predicted_intent: IntentClassifierPrediction = None
        self.file: File = None
        self.chat_history: list[tuple[str, int]] = None

    def reply(self, message: str) -> None:
        """A agent message (usually a reply to a user message) is sent to the session platform to show it to the user.

        Args:
            message (str): the agent reply
        """
        pass


class Platform(ABC):
    """The platform abstract class.

    A platform defines the methods the agent can use to interact with a particular communication channel
    (e.g. Telegram, Slack...) for instance, sending and receiving messages.

    This class serves as a template to implement platforms.
    """

    def __init__(self):
        pass

    def reply(self, session: AgentSession, message: str) -> None:
        """Send a agent reply, i.e. a text message, to a specific user.

        Args:
            session (Session): the user session
            message (str): the message to send to the user
        """
        pass


class WebSocketPlatform(Platform):
    """The WebSocket Platform allows a agent to communicate with the users using the
    `WebSocket <https://en.wikipedia.org/wiki/WebSocket>`_ bidirectional communications protocol.
    """

    def __init__(self):
        super().__init__()

    def reply_file(self, session: AgentSession, file: File) -> None:
        """Send a file reply to a specific user

        Args:
            session (AgentSession): the user session
            file (File): the file to send
        """
        pass

    def reply_dataframe(self, session: AgentSession, df) -> None:
        """Send a DataFrame agent reply, i.e. a table, to a specific user.

        Args:
            session (AgentSession): the user session
            df (pandas.DataFrame): the message to send to the user
        """
        pass

    def reply_options(self, session: AgentSession, options: list[str]):
        """Send a list of options as a reply. They can be used to let the user choose one of them

        Args:
            session (AgentSession): the user session
            options (list[str]): the list of options to send to the user
        """
        pass

    def reply_plotly(self, session: AgentSession, plot) -> None:
        """Send a Plotly figure as a agent reply, to a specific user.

        Args:
            session (AgentSession): the user session
            plot (plotly.graph_objs.Figure): the message to send to the user
        """
        pass

    def reply_location(self, session: AgentSession, latitude: float, longitude: float) -> None:
        """Send a location reply to a specific user.

        Args:
            session (AgentSession): the user session
            latitude (str): the latitude of the location
            longitude (str): the longitude of the location
        """
        pass


class TelegramPlatform(Platform):
    """The Telegram Platform allows a agent to interact via Telegram."""

    def __init__(self):
        super().__init__()

    def reply_file(self, session: AgentSession, file: File, message: str = None) -> None:
        """Send a file reply to a specific user

        Args:
            session (AgentSession): the user session
            file (File): the file to send
            message (str, optional): message to be attached to file, 1024 char limit
        """
        pass

    def reply_image(self, session: AgentSession, file: File, message: str = None) -> None:
        """Send an image reply to a specific user

        Args:
            session (AgentSession): the user session
            file (File): the file to send (the image)
            message (str, optional): message to be attached to file, 1024 char limit
        """
        pass

    def reply_location(self, session: AgentSession, latitude: float, longitude: float) -> None:
        """Send a location reply to a specific user.

        Args:
            session (AgentSession): the user session
            latitude (str): the latitude of the location
            longitude (str): the longitude of the location
        """
        pass


class Entity(NamedElement):
    """Entities are used to specify the type of information to extract from user inputs. These entities are embedded in
    intent parameters.

    Args:
        name (str): Inherited from NamedElement, the entity's name

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the entity.
        visibility (str): Inherited from NamedElement, represents the visibility of the entity.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)


class BaseEntityImpl(Enum):
    """All the available base entities."""

    number = "number_entity"
    datetime = "datetime_entity"
    any = "any_entity"


class BaseEntity(Entity):
    """Predefined entities, which are provided by the agent framework and do not need the user to define them.

    Args:
        name (BaseEntityImpl): the entity's name

    Attributes:
        name (str): Inherited from Entity, represents the name of the entity.
        visibility (str): Inherited from NamedElement, represents the visibility of the entity.
    """

    def __init__(self, name: BaseEntityImpl):
        super().__init__(name.value)


class EntityEntry:
    """Each one of the entries an entity consists of.

    Args:
        value (str): the entry value
        synonyms (list[str] or None): the value synonyms

    Attributes:
        value (str): the entry value
        synonyms (list[str]): The value synonyms
    """

    def __init__(self, value: str, synonyms: list[str]):
        super().__init__()
        self.value: str = value
        self.synonyms: list[str] = synonyms


class CustomEntity(Entity):
    """An entity with custom values.

    Args:
        name (str): the entity's name
        entries (list[EntityEntry]): the entity entries. If base_entity, there are no entries (i.e. None)
        description (str): a description of the entity, optional

    Attributes:
        name (str): Inherited from Entity, represents the name of the entity.
        visibility (str): Inherited from NamedElement, represents the visibility of the entity.
        description (str) : a description of the entity, optional
        entries (list[EntityEntry]): the entity entries
    """

    def __init__(self, name: str, description: str, entries: list[EntityEntry]):
        super().__init__(name)
        self.description: str = description
        self.entries: list[EntityEntry] = entries


class IntentParameter(NamedElement):
    """The intent parameter.

    An intent parameter is composed by a name, a fragment and an entity. The fragment is the intent's training sentence
    substring where an entity should be matched. E.g. in an intent with the training sentence
    "What is the weather in CITY?" we could define a parameter named "city" in the fragment "CITY" that should match
    with any value in the entity "city_entity" (previously defined)

    Args:
        name (str): the intent parameter name
        fragment (str): the fragment the intent's training sentences that is expected to match with the entity
        entity (Entity): the entity to be matched in this parameter

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the intent parameter.
        visibility (str): Inherited from NamedElement, represents the visibility of the intent parameter.
        fragment (str): The fragment the intent's training sentences that is expected to match with the entity
        entity (Entity): The entity to be matched in this parameter
    """

    def __init__(self, name: str, fragment: str, entity: Entity):
        super().__init__(name)
        self.fragment: str = fragment
        self.entity: Entity = entity


class Intent(NamedElement):
    """Intents define the intentions or goals the user can express to a agent.

    An intent is defined by a set of training sentences representing the different ways a user could express an intention
    (e.g. "Hi", "Hello" for a Greetings intent) and/or with a description.

    Intents can also define parameters that are filled with information extracted from the user input using entities.

    Args:
        name (str): the intent's name
        training_sentences (list[str] or None): the intent's training sentences
        parameters (list[IntentParameter] or None): the intent's parameters
        description (str or None): a description of the intent, optional

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the intent
        visibility (str): Inherited from NamedElement, represents the visibility of the intent
        description (str or None): a description of the intent, optional
        training_sentences (list[str]): The intent's training sentences
        parameters (list[IntentParameter]): The intent's parameters
    """

    def __init__(
            self,
            name: str,
            training_sentences: list[str] or None = None,
            parameters: list[IntentParameter] or None = None,
            description: str or None = None
    ):
        super().__init__(name)
        if parameters is None:
            parameters = []
        if training_sentences is None:
            training_sentences = []
        self.description: str = description
        self.training_sentences: list[str] = training_sentences
        self.parameters: list[IntentParameter] = parameters

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def parameter(self, name: str, fragment: str, entity: Entity):
        """Add a parameter to the list of intent parameters.

        Args:
            name (str): The name of the parameter.
            fragment (str): A description or fragment associated with the parameter.
            entity (Entity): The entity that this parameter is related to.

        Returns:
            Intent: Returns the instance of :class:`Intent` it was called on (i.e., self).
        """

        for parameter in self.parameters:
            if parameter.name == name:
                raise ValueError(f"An intent cannot have two parameters with the same name ({name})")
        self.parameters.append(IntentParameter(name, fragment, entity))
        return self


class DummyEvent(Event):
    """Represents a placeholder event."""

    def __init__(self):
        super().__init__(name='dummy_event')


class WildcardEvent(Event):
    """Wildcard event. Can be used to match any event in a transition."""

    def __init__(self):
        super().__init__(name='any_event')


class ReceiveMessageEvent(Event):
    """This event checks if a message is received from the user.

    Args:
        message (str): The message to be checked

    Attributes:
        name (str): Inherited from Event, represents the name of the event.
        visibility (str): Inherited from Event, represents the visibility of the event.
        type (Type): Inherited from Event, represents the type of the event.
        is_abstract (bool): Inherited from Event, indicates if the event is abstract.
        parameters (set[structural.Parameter]): Inherited from Event, the set of parameters for the event.
        owner (Type): Inherited from Event, the type that owns the property.
        code (str): Inherited from Event, code of the event.
    """

    def __init__(self, message: str):
        super().__init__('receive_message')
        self.message: Any = message


class ReceiveTextEvent(ReceiveMessageEvent):
    """Event for receiving text messages. Supports intent prediction.

    Args:
        text (str): the received message content
        session_id (str): the id of the session the event was sent to (can be none)
        human (bool): indicates if the sender is human. Defaults to True

    Attributes:
        _name (str): the name of the event
        predicted_intent (IntentClassifierPrediction): the predicted intent for the event message
    """

    def __init__(self, text: str = None):
        super().__init__(message=text)
        self._name = 'receive_message_text'
        self.predicted_intent: IntentClassifierPrediction = None

    def log(self):
        return f'{self._name} ({self.message})'


class ReceiveJSONEvent(ReceiveMessageEvent):
    """Event for receiving JSON messages.

    Args:
        payload (dict): the received message content
        session_id (str): the id of the session the event was sent to (can be none)
        human (bool): indicates if the sender is human. Defaults to False

    Attributes:
        _name (str): the name of the event
    """

    def __init__(self, payload: dict = None):
        if payload is None:
            payload = {}
        super().__init__(message=json.dumps(payload))
        self._name = 'receive_message_json'


class ReceiveFileEvent(Event):
    """Event for receiving files.

    Args:
        file (File): the received file
        session_id (str): the id of the session the event was sent to (can be none)
        human (bool): indicates if the sender is human. Defaults to True

    Attributes:
        file (File): the received file
        human (bool): indicates if the sender is human. Defaults to True
    """

    def __init__(self, file: File = None):
        super().__init__(name='receive_file')
        self.file: File = file


class IntentMatcher(Condition):
    """This event checks if 2 intents are the same (returning True, and False otherwise), used for intent matching
    checking.

    Args:
        intent (Intent): The reference intent to compare with the target intent

    Attributes:
        name (str): Inherited from Event, represents the name of the event.
        visibility (str): Inherited from Event, represents the visibility of the event.
        type (Type): Inherited from Event, represents the type of the event.
        is_abstract (bool): Inherited from Event, indicates if the event is abstract.
        parameters (set[structural.Parameter]): Inherited from Event, the set of parameters for the event.
        owner (Type): Inherited from Event, the type that owns the property.
        code (str): Inherited from Event, code of the event.
        intent (Intent): The reference intent to compare with the target intent
    """

    def __init__(self, intent: Intent):
        super().__init__('intent_matched', None)
        self.intent: Intent = intent


class VariableOperationMatcher(Condition):
    """This event checks if for a specific comparison operation, using a stored session value
    and a given target value, returns true (e.g., 'temperature' > 30, where var_name = 'temperature',
    operation = `op.greater` and target = 30)

    Args:
        var_name (str): The variable name (stored in the user session)
        operation (Callable[[Any, Any], bool]): The operation function
        target (Any): The target value

    Attributes:
        name (str): Inherited from Event, represents the name of the event.
        visibility (str): Inherited from Event, represents the visibility of the event.
        type (Type): Inherited from Event, represents the type of the event.
        is_abstract (bool): Inherited from Event, indicates if the event is abstract.
        parameters (set[structural.Parameter]): Inherited from Event, the set of parameters for the event.
        owner (Type): Inherited from Event, the type that owns the property.
        code (str): Inherited from Event, code of the event.
        var_name (str): The variable name (stored in the user session)
        operation (Callable[[Any, Any], bool]): The operation function
        target (Any): The target value
    """

    def __init__(self, var_name: str, operation: Callable[[Any, Any], bool], target: Any):
        super().__init__('variable_matches_operation', None)
        self.var_name: str = var_name
        self.operation: Callable[[Any, Any], bool] = operation
        self.target: Any = target


class FileTypeMatcher(Condition):
    """This event only returns True if a user just sent a file.

    Args:
        allowed_types (list[str]): The file types that will be considered in the event

    Attributes:
        name (str): Inherited from Event, represents the name of the event.
        visibility (str): Inherited from Event, represents the visibility of the event.
        type (Type): Inherited from Event, represents the type of the event.
        is_abstract (bool): Inherited from Event, indicates if the event is abstract.
        parameters (set[structural.Parameter]): Inherited from Event, the set of parameters for the event.
        owner (Type): Inherited from Event, the type that owns the property.
        code (str): Inherited from Event, code of the event.
        allowed_types (list[str]): The file types that will be considered in the event
    """

    def __init__(self, allowed_types: list[str] or str = None):
        super().__init__('file_matches_types', None)
        self.allowed_types: list[str] or str = allowed_types


# should this not be an event?
class Auto(Condition):
    """This condition always returns True.

    Attributes:
        name (str): Inherited from Event, represents the name of the event.
        visibility (str): Inherited from Event, represents the visibility of the event.
        type (Type): Inherited from Event, represents the type of the event.
        is_abstract (bool): Inherited from Event, indicates if the event is abstract.
        parameters (set[structural.Parameter]): Inherited from Event, the set of parameters for the event.
        owner (Type): Inherited from Event, the type that owns the property.
        code (str): Inherited from Event, code of the event.
    """

    def __init__(self):
        super().__init__('auto', None)


class AgentState(State):
    """A agent state.

    Args:
        agent (Agent): the agent the state belongs to
        name (str): the state name
        initial (bool): whether the state is initial or not

    Attributes:
        name (str): Inherited from State, the state name
        visibility (str): Inherited from State, determines the kind of visibility of the state (public as default).
        sm (StateMachine): Inherited from State, the state machine the state belongs to
        initial (bool): Inherited from State, whether the state is initial or not
        transitions (list[Transition]): Inherited from State, the state's transitions to other states
        body (Body): Inherited from State, the body of the state
        fallback_body (Body): Inherited from State, the fallback body of the state
        _transition_counter (int): Inherited from State, count the number of transitions of this state. Used to name the transitions.
    """

    def __init__(
            self,
            agent: 'Agent',
            name: str,
            initial: bool = False,
            ic_config: IntentClassifierConfiguration or None = None
    ):
        super().__init__(agent, name, initial)
        self.agent: 'Agent' = agent
        self.ic_config: IntentClassifierConfiguration = ic_config
        self.intents: list[Intent] = []

    def set_global(self, intent: Intent):
        """Set state as globally accessible state.

        Args:
            intent (Intent): the intent that should trigger the jump to the global state
        """
        self.agent.global_initial_states.append((self, intent))

    def go_to(self, dest: 'AgentState') -> None:
        """Create a new `auto` transition on this state.

        This transition needs no event to be triggered, which means that when the agent moves to a state 
        that has an `auto` transition, the agent will move to the transition's destination state 
        unconditionally without waiting for user input. This transition cannot be combined with other
        transitions.

        Args:
            dest (AgentState): the destination state
        """
        transition_builder: TransitionBuilder = TransitionBuilder(source=self, event=None, conditions=Auto())
        transition_builder.go_to(dest)

    def when_intent_matched(self, intent: Intent) -> TransitionBuilder:
        """Start the definition of an "intent matching" transition on this state.

        Args:
            intent (Intent): the target intent for the transition to be triggered

        Returns:
            TransitionBuilder: the transition builder
        """
        if intent in self.intents:
            raise ValueError(f"Duplicated intent matching transition in a state ({intent.name}).")
        if intent not in self.agent.intents:
            raise ValueError(f"Intent {intent.name} not found")
        self.intents.append(intent)
        event: ReceiveTextEvent = ReceiveTextEvent()
        condition: Condition = IntentMatcher(intent)
        transition_builder: TransitionBuilder = TransitionBuilder(source=self, event=event, conditions=condition)
        return transition_builder

    def when_no_intent_matched(self) -> TransitionBuilder:
        event: ReceiveTextEvent = ReceiveTextEvent()
        condition: Condition = IntentMatcher(Intent("fallback_intent"))
        transition_builder: TransitionBuilder = TransitionBuilder(source=self, event=event, conditions=condition)
        return transition_builder

    def when_variable_matches_operation(
            self,
            var_name: str,
            operation: Callable[[Any, Any], bool],
            target: Any,
    ) -> TransitionBuilder:
        """Start the definition of a "variable matching operator" transition on this state.

        This transition evaluates if (variable operator target_value) is satisfied. For instance, "age > 18".

        Args:
            var_name (str): the name of the variable to evaluate. The variable must exist in the user session
            operation (Callable[[Any, Any], bool]): the operation to apply to the variable and the target value. It
                gets as arguments the variable and the target value, and returns a boolean value
            target (Any): the target value to compare with the variable

        Returns:
            TransitionBuilder: the transition builder
        """
        condition: Condition = VariableOperationMatcher(var_name, operation, target)
        transition_builder: TransitionBuilder = TransitionBuilder(source=self, conditions=condition)
        return transition_builder

    def when_file_received(self, allowed_types: list[str] or str = None) -> TransitionBuilder:
        """Start the definition of a "file received" transition on this state.

        Args:
            allowed_types (list[str] or str): the file types to consider for this transition. List of strings or just 1
                string are valid values

        Returns:
            TransitionBuilder: the transition builder
        """
        event = ReceiveFileEvent()
        transition_builder: TransitionBuilder = TransitionBuilder(source=self, event=event, conditions=FileTypeMatcher(allowed_types))
        return transition_builder


class Agent(StateMachine):
    """A agent model.

    Args:
        name (str): the agent name

    Attributes:
        name (str): Inherited from StateMachine, represents the name of the agent.
        visibility (str): Inherited from StateMachine, determines the kind of visibility of the agent (public as default).
        states (list[state_machine.State]): Inherited from StateMachine, the states of the agent
        properties (list[ConfigProperty]): Inherited from StateMachine, the configuration properties of the agent.
        platforms (list[Platform]): The agent platforms.
        default_ic_config (IntentClassifierConfiguration): the intent classifier configuration used by default for the
            agent states.
        intents (list[Intent]): The agent intents.
        entities (list[Entity]): The agent entities.
        global_initial_states (list[state_machine.State, Intent]): List of tuples of initial global states and their triggering intent
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.platforms: list[Platform] = []
        self.default_ic_config: IntentClassifierConfiguration = None
        self.intents: list[Intent] = []
        self.entities: list[Entity] = []
        self.global_initial_states: list[tuple[AgentState, Intent]] = []
        self.llms: list[LLMWrapper] = []
        self.rags: list[RAG] = []

    def new_state(self,
                  name: str,
                  initial: bool = False,
                  ic_config: IntentClassifierConfiguration or None = None
                  ) -> AgentState:
        """Create a new state in the agent.

        Args:
            name (str): the state name. It must be unique in the agent.
            initial (bool): whether the state is initial or not. A agent must have 1 initial state.
            ic_config (IntentClassifierConfiguration or None): the intent classifier configuration for the state.
                If None is provided, the agent's default one will be assigned to the state.

        Returns:
            AgentState: the state
        """
        new_state = AgentState(self, name, initial, ic_config)
        if new_state in self.states:
            raise ValueError(f"Duplicated state in agent ({new_state.name})")
        if initial and self.initial_state():
            raise ValueError(f"A agent must have exactly 1 initial state")
        if not initial and not self.states:
            raise ValueError(f"The first state of a agent must be initial")
        self.states.append(new_state)
        return new_state

    def add_intent(self, intent: Intent) -> Intent:
        """Add an intent to the agent.

        Args:
            intent (Intent): the intent to add

        Returns:
            Intent: the added intent
        """
        if intent in self.intents:
            raise ValueError(f"A agent cannot have two intents with the same name ({intent.name}).")
        self.intents.append(intent)
        return intent

    def new_intent(self,
                   name: str,
                   training_sentences: list[str] or None = None,
                   parameters: list[IntentParameter] or None = None,
                   description: str or None = None,
                   ) -> Intent:
        """Create a new intent in the agent.

        Args:
            name (str): the intent name. It must be unique in the agent
            training_sentences (list[str] or None): the intent's training sentences
            parameters (list[IntentParameter] or None): the intent parameters, optional
            description (str or None): a description of the intent, optional

        Returns:
            Intent: the intent
        """
        new_intent = Intent(name, training_sentences, parameters, description)
        if new_intent in self.intents:
            raise ValueError(f"A agent cannot have two intents with the same name ({new_intent.name}).")
        self.intents.append(new_intent)
        return new_intent

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the agent.

        Args:
            entity (Entity): the entity to add

        Returns:
            Entity: the added entity
        """
        if entity in self.entities:
            raise ValueError(f"A agent cannot have two entities with the same name ({entity.name}).")
        self.entities.append(entity)
        return entity

    def new_entity(self,
                   name: BaseEntityImpl or str,
                   base_entity: bool = False,
                   entries: dict[str, list[str]] or None = None,
                   description: str or None = None
                   ) -> Entity:
        """Create a new entity in the agent.

        Args:
            name (str): the entity name. It must be unique in the agent
            base_entity (bool): whether the entity is a base entity or not (i.e. a custom entity)
            entries (dict[str, list[str]] or None): the entity entries
            description (str or None): a description of the entity, optional

        Returns:
            Entity: the entity
        """
        if base_entity:
            new_entity = BaseEntity(name)
        else:
            entity_entries = []
            for value, synonyms in entries.items():
                entity_entries.append(EntityEntry(value, synonyms))
            new_entity = CustomEntity(name, description, entity_entries)
        if new_entity in self.entities:
            raise ValueError(f"A agent cannot have two entities with the same name ({new_entity.name}).")
        self.entities.append(new_entity)
        return new_entity

    def new_rag(
            self,
            name: str,
            vector_store: RAGVectorStore,
            splitter: RAGTextSplitter,
            llm_name: str,
            k: int = 4,
            num_previous_messages: int = 0,
    ) -> RAG:
        """Register a Retrieval-Augmented Generation configuration on the agent."""

        if any(existing.name == name for existing in self.rags):
            raise ValueError(f"A agent cannot have two RAG configurations with the same name ({name}).")
        rag = RAG(
            name=name,
            agent=self,
            vector_store=vector_store,
            splitter=splitter,
            llm_name=llm_name,
            k=k,
            num_previous_messages=num_previous_messages,
        )
        self.rags.append(rag)
        return rag

    def use_websocket_platform(self) -> WebSocketPlatform:
        """Use the WebSocketPlatform on this agent.

        Returns:
            WebSocketPlatform: the websocket platform
        """
        websocket_platform = WebSocketPlatform()
        for platform in self.platforms:
            if isinstance(platform, WebSocketPlatform):
                return None  # Only 1 platform max of each kind
        self.platforms.append(websocket_platform)
        return websocket_platform

    def use_telegram_platform(self) -> TelegramPlatform:
        """Use the TelegramPlatform on this agent.

        Returns:
            TelegramPlatform: the telegram platform
        """
        telegram_platform = TelegramPlatform()
        for platform in self.platforms:
            if isinstance(platform, TelegramPlatform):
                return None  # Only 1 platform max of each kind
        self.platforms.append(telegram_platform)
        return telegram_platform


class MatchedParameter:
    """A matched parameter in a user input (i.e. an entity that is found in a user message, which is an intent
    parameter).

    Args:
        name (str): the parameter name
        value (object or None): the parameter value
        info (dict or None): extra parameter information

    Attributes:
        name (str): The parameter name
        value (object or None): The parameter value
        info (dict): Extra parameter information
    """

    def __init__(self,
                 name: str,
                 value: object or None = None,
                 info: dict or None = None):
        if info is None:
            info = {}
        self.name: str = name
        self.value: object or None = value
        self.info: dict = info


class IntentClassifierPrediction:
    """The prediction result of an Intent Classifier for a specific intent.

    The intent classifier tries to determine the intent of a user message. For each possible intent, it will return
    an IntentClassifierPrediction containing the results, that include the probability itself and other information.

    Args:
        intent (Intent): the target intent of the prediction
        score (float): the probability that this is the actual intent of the user message
        matched_sentence (str): the sentence used in the intent classifier (the original user message is previously
            processed, is modified by the NER, etc.)
        matched_parameters (list[MatchedParameter]): the list of parameters (i.e. entities) found in the user message

    Attributes:
        intent (Intent): The target intent of the prediction
        score (float): The probability that this is the message intent
        matched_sentence (str): The sentence used in the intent classifier (the original user message is previously
            processed, is modified by the NER, etc.)
        matched_parameters (list[MatchedParameter]): The list of parameters (i.e. entities) found in the user message
    """

    def __init__(
            self,
            intent: Intent,
            score: float = None,
            matched_sentence: str = None,
            matched_parameters: list[MatchedParameter] = None
    ):
        self.intent: Intent = intent
        self.score: float = score
        self.matched_sentence: str = matched_sentence
        self.matched_parameters: list[MatchedParameter] = matched_parameters

    def get_parameter(self, name: str) -> MatchedParameter or None:
        """Get a parameter from the intent classifier prediction.

        Args:
            name (str): the name of the parameter to get

        Returns:
            MatchedParameter or None: the parameter if it exists, None otherwise
        """
        for parameter in self.matched_parameters:
            if parameter.name == name:
                return parameter
        return None
