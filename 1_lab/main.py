from __future__ import annotations
from threading import Thread
from typing import List
from matplotlib import pyplot as plt
from enum import Enum
from time import time
import numpy as np
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple
from numpy import random as rnd
from time import time, sleep


def get_current_time() -> float:
    return time()


class Protocol(Enum):
    kGbn = 0,
    kSrp = 1

    @staticmethod
    def to_str(protocol: Protocol) -> str:
        if protocol == Protocol.kGbn:
            return 'Go-Back-N'
        elif protocol == Protocol.kSrp:
            return 'Selective Repeat'

        return ''

    @staticmethod
    def to_short_str(protocol: Protocol) -> str:
        if protocol == Protocol.kGbn:
            return 'GBN'
        elif protocol == Protocol.kSrp:
            return 'SRP'

        return ''


class MsgCode(Enum):
    kSuccess = 0,
    kFail = 1


class Message:
    def __init__(self, id: int, code: MsgCode, data) -> None:
        self.id = id
        self.code = code
        self.data = data


class SlidingWindowProtocol(ABC):
    @staticmethod
    def connect(sender: SlidingWindowSender, receiver: SlidingWindowReceiver) -> None:
        sender.connect(receiver)
        receiver.connect(sender)

    def create_connected(
            type: Protocol,
            window_size: int,
            timeout: float,
            corruption_rate: float) -> Tuple[SlidingWindowSender, SlidingWindowReceiver]:
        sender: SlidingWindowSender = None
        receiver: SlidingWindowReceiver = None

        if type == Protocol.kGbn:
            sender = GoBackNSender(window_size, timeout)
            receiver = GoBackNReceiver(corruption_rate)
        elif type == Protocol.kSrp:
            sender = SelectiveRepeatSender(window_size, timeout)
            receiver = SelectiveReapetReceiver(corruption_rate)
        else:
            assert False

        SlidingWindowProtocol.connect(sender, receiver)
        return sender, receiver

    def __init__(self) -> None:
        super().__init__()
        self.message_queue: List[Message] = []

    def get_message(self, message: Message) -> None:
        self.message_queue.append(message)

    @abstractmethod
    def run(self) -> None:
        pass


class SlidingWindowSender(SlidingWindowProtocol):
    def __init__(self, window_size: int, timeout_time: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.receiver: SlidingWindowReceiver = None
        self.timeout_time = timeout_time
        self.max_messages_num = 100
        self.finished = False
        self.message_counter = 0

    def connect(self, receiver: SlidingWindowReceiver) -> None:
        assert receiver is not None
        self.receiver = receiver

    def set_max_messages_num(self, messages_num: int) -> None:
        assert messages_num > 0
        self.max_messages_num = messages_num

    def send_message_to_receiver(self, message: Message) -> None:
        self.message_counter += 1
        self.receiver.get_message(message)

    def is_finished(self) -> bool:
        return self.finished

    def get_sended_messages_num(self) -> int:
        return self.message_counter


class SlidingWindowReceiver(SlidingWindowProtocol):
    def __init__(self, corruption_rate: float) -> None:
        super().__init__()
        self.sender: SlidingWindowSender = None
        self.corruption_rate = max(min(corruption_rate, 1.0), 0.0)
        self.rnd = rnd.default_rng()

    def connect(self, sender: SlidingWindowSender) -> None:
        assert sender is not None
        self.sender = sender

    def prepare_answer(self, message_id: int) -> Message:
        corruption_rnd = self.rnd.uniform(0.0, 1.0)

        if corruption_rnd < self.corruption_rate:
            # corrupted
            return Message(message_id, MsgCode.kFail, None)

        # success
        return Message(message_id, MsgCode.kSuccess, None)


class GoBackNSender(SlidingWindowSender):
    def __init__(self, window_size: int, timeout_time: float) -> None:
        super().__init__(window_size, timeout_time)
        self.send_base = 0
        self.send_base_time = get_current_time()
        self.send_next = 0
        self.message_id = 0
        self.waiting_message_id = 0
        self.dummy_data = 'go back n sender data'

    def run(self) -> None:
        # print('sender start work')

        while self.send_base < self.max_messages_num:
            handle_error = False

            if (self.send_next - self.send_base < self.window_size) and (self.send_next < self.max_messages_num):
                send_message = Message(self.message_id, MsgCode.kSuccess, self.dummy_data)
                self.send_message_to_receiver(send_message)

                self.send_base_time = get_current_time()
                self.send_next += 1
                self.message_id += 1
                # print(f'sender send id {send_message.id}')

            if self.remove_outdated_messages() > 0:
                # print(f'message info, id = {self.message_queue[0].id}, waiting = {self.waiting_message_id}')

                if self.message_queue[0].id == self.waiting_message_id and self.message_queue[0].code == MsgCode.kSuccess:
                    self.send_base += 1
                    self.waiting_message_id += 1
                    del self.message_queue[0]
                    # print(f'sender move window, new base {self.send_base}')
                else:
                    handle_error = True

            if handle_error or self.time_since_base_send() > self.timeout_time:
                # print(f'moved back to {self.send_next - self.send_base}')
                self.send_next = self.send_base
                self.waiting_message_id = self.message_id

        self.finished = True
        # print(f'GBN total sended: {self.message_counter}')

    def remove_outdated_messages(self) -> int:
        timouted_messages = 0
        queue_size = len(self.message_queue)

        if queue_size > 0:
            while timouted_messages < queue_size and self.message_queue[timouted_messages].id < self.waiting_message_id:
                timouted_messages += 1

            if timouted_messages > 0:
                # print(f'outdated removed: {timouted_messages}')
                del self.message_queue[0:timouted_messages]
                # print(f'head = {self.message_queue[0].id if len(self.message_queue) > 0 else -1}, wating = {self.waiting_message_id}')

        return queue_size - timouted_messages

    def time_since_base_send(self) -> float:
        return get_current_time() - self.send_base_time


class GoBackNReceiver(SlidingWindowReceiver):
    def __init__(self, corruption_rate: float) -> None:
        super().__init__(corruption_rate)
        self.last_received = 0

    def run(self) -> None:
        # print('receiver start work')

        while not self.sender.is_finished():
            if len(self.message_queue) > 0:
                current_message = self.message_queue[0]

                send_message = self.prepare_answer(current_message.id)
                self.sender.get_message(send_message)

                del self.message_queue[0]
                # print(f'receiver receive={current_message.id}')


class SelectiveRepeatSender(SlidingWindowSender):
    class MessageNode:
        def __init__(self, send_time: float, message: Message) -> None:
            self.send_time = send_time
            self.message = message

    def __init__(self, window_size: int, timeout_time: float) -> None:
        super().__init__(window_size, timeout_time)
        self.last_approved = 0
        self.send_next = 0
        self.message_nodes: Dict[int, SelectiveRepeatSender.MessageNode] = {}
        self.dummy_data = 'selective repeat dummy data'

    def run(self) -> None:
        while self.finished == False:
            queue_size = len(self.message_queue)
            while queue_size > 0:
                message = self.message_queue[0]
                self.last_approved = max(self.last_approved, message.data)

                if self.last_approved >= self.max_messages_num - 1:
                    self.finished = True
                    break

                if message.id in self.message_nodes:
                    if message.code == MsgCode.kSuccess:
                        del self.message_nodes[message.id]
                    else:
                        self.send_message_to_receiver(self.message_nodes[message.id].message)
                        self.message_nodes[message.id].send_time = get_current_time()

                del self.message_queue[0]
                queue_size -= 1

            if len(self.message_nodes) < self.window_size and self.send_next < self.max_messages_num:
                send_message = Message(self.send_next, MsgCode.kSuccess, self.dummy_data)

                self.send_message_to_receiver(send_message)
                self.message_nodes[self.send_next] = SelectiveRepeatSender.MessageNode(get_current_time(), send_message)

                self.send_next += 1

            for message_node_idx in self.message_nodes:
                message_node = self.message_nodes[message_node_idx]

                if self.is_outdated(message_node.send_time):
                    self.send_message_to_receiver(message_node.message)
                    self.message_nodes[message_node.message.id].send_time = get_current_time()
                    # print(f'repeat outdated {message_node.message.id}')

        # print(f'SRP total sended: {self.message_counter}')

    def is_outdated(self, send_time: float) -> bool:
        # print(f'current = {get_current_time()}, send = {send_time}')
        return get_current_time() - send_time > self.timeout_time


class SelectiveReapetReceiver(SlidingWindowReceiver):
    def __init__(self, corruption_rate: float) -> None:
        super().__init__(corruption_rate)
        self.last_received = -1
        self.received: List[int] = []

    def run(self) -> None:
        info_send_time = get_current_time()

        while not self.sender.is_finished():
            self.resolve_last_received()

            if len(self.message_queue) > 0:
                current_message = self.message_queue[0]

                send_message = self.prepare_answer(current_message.id)
                send_message.data = self.last_received
                self.sender.get_message(send_message)

                if send_message.code == MsgCode.kSuccess:
                    self.received.append(send_message.id)

                del self.message_queue[0]

            elif get_current_time() - info_send_time > 0.01:  # exta safe
                info_send_time = get_current_time()
                self.sender.get_message(Message(-1, MsgCode.kSuccess, self.last_received))

    def resolve_last_received(self) -> None:
        i = 0
        while i < len(self.received):
            if self.received[i] <= self.last_received:
                del self.received[i]
                i -= 1

            elif self.received[i] == self.last_received + 1:
                self.last_received += 1

            i += 1


kMessagesToSend = 100
kThisFilePath = os.path.abspath(__file__)

def img_save_dst() -> str:
    return 'report/img/'


class Statistics(Enum):
    kMessageNum = 0,
    kWorkingTime = 1

    @staticmethod
    def to_str(stat: Statistics) -> str:
        if stat == Statistics.kMessageNum:
            return 'MessageNum'
        elif stat == Statistics.kWorkingTime:
            return 'WorkingTime'
        
        return ''


def run_protocol(sender: SlidingWindowSender, receiver: SlidingWindowReceiver) -> None:
    sender_thread = Thread(target=sender.run)
    receiver_thread = Thread(target=receiver.run)

    sender_thread.start()
    receiver_thread.start()

    sender_thread.join()
    receiver_thread.join()


def calculate_corruption_rate_dependencies(
        protocol_type: Protocol,
        window_size: int,
        timeout: float,
        corruption_rates: List[float],
        stat: Statistics,
        show_plot: bool = True) -> None:
    messages_nums = []
    work_times = []

    for corruption_rate in corruption_rates:
        sender, receiver = SlidingWindowProtocol.create_connected(protocol_type, window_size, timeout, corruption_rate)
        sender.set_max_messages_num(kMessagesToSend)

        start_time = time()
        run_protocol(sender, receiver)
        work_time = time() - start_time

        print(f'corruption_rate = {corruption_rate}, message_num = {sender.get_sended_messages_num()}, work_time = {work_time}')
        messages_nums.append(sender.get_sended_messages_num())
        work_times.append(work_time)

    plt.plot(corruption_rates, messages_nums if stat == Statistics.kMessageNum else work_times, label=f'window size = {window_size}')
    if show_plot:
        plt.legend()
        plt.show()


def calculate_window_size_dependencies(
        protocol_type: Protocol,
        timeout: float,
        corruption_rate: float,
        window_sizes: List[int],
        stat: Statistics,
        show_plot: bool = True) -> None:
    messages_nums = []
    work_times = []

    for window_size in window_sizes:
        sender, receiver = SlidingWindowProtocol.create_connected(protocol_type, window_size, timeout, corruption_rate)
        sender.set_max_messages_num(kMessagesToSend)

        start_time = time()
        run_protocol(sender, receiver)
        work_time = time() - start_time

        print(f'window_size = {window_size}, message_num = {sender.get_sended_messages_num()}, work_time = {work_time}')
        messages_nums.append(sender.get_sended_messages_num())
        work_times.append(work_time)

    plt.plot(window_sizes, messages_nums if stat == Statistics.kMessageNum else work_times, label=f'corruption rate = {corruption_rate}')
    if show_plot:
        plt.legend()
        plt.show()


def calculate_timeout_dependencies(
        protocol_type: Protocol,
        window_size: int,
        corruption_rate: float,
        timeouts: List[float],
        stat: Statistics,
        show_plot: bool = True) -> None:
    messages_nums = []
    work_times = []

    for timeout in timeouts:
        sender, receiver = SlidingWindowProtocol.create_connected(protocol_type, window_size, timeout, corruption_rate)
        sender.set_max_messages_num(kMessagesToSend)

        start_time = time()
        run_protocol(sender, receiver)
        work_time = time() - start_time

        print(f'window_size = {window_size}, message_num = {sender.get_sended_messages_num()}, work_time = {work_time}')
        messages_nums.append(sender.get_sended_messages_num())
        work_times.append(work_time)

    plt.plot(timeouts, messages_nums if stat == Statistics.kMessageNum else work_times, label=Protocol.to_str(protocol_type))
    if show_plot:
        plt.legend()
        plt.show()



def calculate_size_rate_dependencies(
        protocol_type: Protocol,
        timeout: float,
        window_sizes: List[int],
        corruption_rates: List[float],
        stat: Statistics) -> None:
    
    for window_size in window_sizes:
        calculate_corruption_rate_dependencies(protocol_type, window_size, timeout, corruption_rates, stat, False)

    plt.legend()
    plt.xlabel('corruption rate')
    plt.ylabel('total messages send' if stat == Statistics.kMessageNum else 'working time (in seconds)')
    plt.title(f'{Protocol.to_str(protocol_type)}')
    plt.savefig(f'{img_save_dst()}sizeRate{Protocol.to_short_str(protocol_type)}{Statistics.to_str(stat)}.png')
    plt.clf()


def calculate_rate_size_dependencies(
        protocol_type: Protocol,
        timeout: float,
        corruption_rates: List[float],
        window_sizes: List[int],
        stat: Statistics) -> None:
    
    for corruption_rate in corruption_rates:
        calculate_window_size_dependencies(protocol_type, timeout, corruption_rate, window_sizes, stat, False)

    plt.legend()
    plt.xlabel('window size')
    plt.ylabel('total messages send' if stat == Statistics.kMessageNum else 'working time (in seconds)')
    plt.title(f'{Protocol.to_str(protocol_type)}')
    plt.savefig(f'{img_save_dst()}rateSize{Protocol.to_short_str(protocol_type)}{Statistics.to_str(stat)}.png')
    plt.clf()


def calculate_protocol_timeout_dependencies(
        window_size: int,
        corruption_rate: float,
        protocol_types: List[Protocol],
        timeouts: List[float],
        stat: Statistics) -> None:
    
    for protocol_type in protocol_types:
        calculate_timeout_dependencies(protocol_type, window_size, corruption_rate, timeouts, stat, False)

    plt.legend()
    plt.xlabel('timeout (in seconds)')
    plt.ylabel('total messages send' if stat == Statistics.kMessageNum else 'working time (in seconds)')
    plt.title('')
    plt.savefig(f'{img_save_dst()}timeouts{Statistics.to_str(stat)}.png')
    plt.clf()


def main():
    corruption_rates = np.linspace(0.0, 0.9, 19)
    window_sizes = [int(x) for x in np.linspace(5, 50, 10)]
    timeouts = np.linspace(0.02, 0.5, 30)

    calculate_size_rate_dependencies(Protocol.kSrp, 0.5, [10, 25, 50], corruption_rates, Statistics.kMessageNum)
    calculate_rate_size_dependencies(Protocol.kSrp, 0.5, [0.1, 0.25, 0.5], window_sizes, Statistics.kMessageNum)

    calculate_size_rate_dependencies(Protocol.kGbn, 0.5, [10, 25, 50], corruption_rates, Statistics.kMessageNum)
    calculate_rate_size_dependencies(Protocol.kGbn, 0.5, [0.1, 0.25, 0.5], window_sizes, Statistics.kMessageNum)

    calculate_size_rate_dependencies(Protocol.kSrp, 0.5, [10, 25, 50], corruption_rates, Statistics.kWorkingTime)
    calculate_rate_size_dependencies(Protocol.kSrp, 0.5, [0.1, 0.25, 0.5], window_sizes, Statistics.kWorkingTime)

    calculate_size_rate_dependencies(Protocol.kGbn, 0.5, [10, 25, 50], corruption_rates, Statistics.kWorkingTime)
    calculate_rate_size_dependencies(Protocol.kGbn, 0.5, [0.1, 0.25, 0.5], window_sizes, Statistics.kWorkingTime)

    calculate_protocol_timeout_dependencies(10, 0.0, [Protocol.kGbn, Protocol.kSrp], timeouts, Statistics.kMessageNum)
    calculate_protocol_timeout_dependencies(10, 0.0, [Protocol.kGbn, Protocol.kSrp], timeouts, Statistics.kWorkingTime)

    return


if __name__ == '__main__':
    main()
